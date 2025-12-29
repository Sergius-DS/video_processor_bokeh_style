import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List
from collections import deque
from enum import Enum
import time
import os


class BlurMode(Enum):
    """Blur modes like Zoom's options"""
    BLUR = "blur"
    REPLACE = "replace"


@dataclass
class QualityPreset:
    """Quality presets matching Zoom's options"""
    name: str
    blur_radius: int
    edge_refinement: int
    temporal_frames: int
    description: str


PRESETS: Dict[str, QualityPreset] = {
    "low": QualityPreset("Low", 15, 5, 3, "Fast processing, lighter blur"),
    "medium": QualityPreset("Medium", 21, 7, 4, "Balanced quality and speed"),
    "high": QualityPreset("High", 27, 9, 5, "Best quality, stronger blur"),
}


class ZoomStyleBokeh:
    """
    Production bokeh processor approaching Zoom/Teams quality.

    Features matching Zoom/Teams:
    - Quality presets (Low/Medium/High)
    - False positive filtering
    - Temporal consistency
    - Edge-aware alpha matting
    - Optional background replacement
    - Performance monitoring
    """

    def __init__(self,
                 model_path: str = 'models/selfie_segmenter.tflite',
                 preset: str = "high",
                 mode: BlurMode = BlurMode.BLUR,
                 background_image: Optional[str] = None):

        self.seg_options = vision.ImageSegmenterOptions(
            base_options=python.BaseOptions(model_asset_path=model_path),
            output_category_mask=False,
            output_confidence_masks=True
        )

        self.preset = PRESETS.get(preset, PRESETS["high"])
        self.mode = mode
        self.background_image = None

        if background_image and os.path.exists(background_image):
            self.background_image = cv2.imread(background_image)

        # ============================================
        # SEGMENTATION THRESHOLDS
        # ============================================
        self.base_threshold = 0.08
        self.protection_threshold = 0.12
        self.edge_threshold = 0.03

        # ============================================
        # REGION-SPECIFIC DILATION
        # ============================================
        self.head_dilate = (7, 5)
        self.shoulder_dilate = (30, 9)
        self.body_dilate = (20, 12)
        self.right_extend = 25
        self.left_extend = 15

        # ============================================
        # FALSE POSITIVE FILTERING
        # ============================================
        self.min_person_area_ratio = 0.05
        self.max_person_area_ratio = 0.85
        self.min_aspect_ratio = 0.3
        self.max_aspect_ratio = 4.0
        self.min_contour_area = 5000
        self.edge_margin = 50

        # ============================================
        # TEMPORAL STABILITY
        # ============================================
        self.mask_history = deque(maxlen=self.preset.temporal_frames)
        self.prev_frame_gray = None
        self.prev_mask = None
        self.motion_threshold = 0.02

        # ============================================
        # TWO-PASS STATE
        # ============================================
        self.best_mask = None
        self.best_quality = 0.0
        self.best_frame_idx = -1
        self.accumulated_mask = None
        self.person_center = None
        self.person_bbox = None

        # ============================================
        # PERFORMANCE TRACKING
        # ============================================
        self.frame_times = []
        self.frame_count = 0

    # ============================================
    # FALSE POSITIVE FILTERING
    # ============================================

    def _filter_false_positives(self, mask: np.ndarray, raw_conf: np.ndarray,
                                 h: int, w: int) -> np.ndarray:
        """Filter out non-person detections"""
        binary = (mask > 0.5).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return mask

        frame_area = h * w
        scored_contours = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_contour_area:
                continue

            x, y, bw, bh = cv2.boundingRect(contour)
            area_ratio = area / frame_area
            aspect_ratio = bh / bw if bw > 0 else 0

            # Center of contour
            M = cv2.moments(contour)
            cx = int(M["m10"] / M["m00"]) if M["m00"] > 0 else x + bw // 2
            cy = int(M["m01"] / M["m00"]) if M["m00"] > 0 else y + bh // 2

            # Edge check
            at_edge = (x < self.edge_margin or y < self.edge_margin or
                      x + bw > w - self.edge_margin or y + bh > h - self.edge_margin)

            # Center bias
            dist = np.sqrt((cx - w/2)**2 + (cy - h/2)**2)
            max_dist = np.sqrt((w/2)**2 + (h/2)**2)
            center_score = 1.0 - (dist / max_dist)

            # Confidence in region
            contour_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(contour_mask, [contour], -1, 255, -1)
            avg_conf = raw_conf[contour_mask > 0].mean() if contour_mask.sum() > 0 else 0

            # Scoring
            score = 0.0
            if self.min_person_area_ratio <= area_ratio <= self.max_person_area_ratio:
                score += 0.3
            if self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio:
                score += 0.2
            score += 0.2 * center_score
            score += 0.2 * avg_conf
            if at_edge and area_ratio < 0.1:
                score -= 0.3
            if aspect_ratio >= 0.5:
                score += 0.1

            scored_contours.append({
                'contour': contour,
                'area': area,
                'center': (cx, cy),
                'bbox': (x, y, bw, bh),
                'confidence': avg_conf,
                'score': score
            })

        if not scored_contours:
            return np.zeros((h, w), dtype=np.float32)

        # Keep best contour
        scored_contours.sort(key=lambda x: (x['score'], x['area']), reverse=True)
        best = scored_contours[0]

        self.person_center = best['center']
        self.person_bbox = best['bbox']

        # Create filtered mask
        filtered = np.zeros((h, w), dtype=np.float32)
        cv2.drawContours(filtered, [best['contour']], -1, 1.0, -1)

        # Include nearby high-confidence contours (e.g., raised hand)
        for other in scored_contours[1:]:
            dist = np.sqrt((other['center'][0] - best['center'][0])**2 +
                          (other['center'][1] - best['center'][1])**2)
            if dist < max(best['bbox'][2], best['bbox'][3]) * 0.5 and other['confidence'] > 0.3:
                cv2.drawContours(filtered, [other['contour']], -1, 1.0, -1)

        return mask * filtered

    # ============================================
    # MASK GENERATION
    # ============================================

    def _get_bbox(self, mask: np.ndarray) -> Optional[dict]:
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        return {'x': x, 'y': y, 'w': w, 'h': h}

    def _extend_shoulders(self, mask: np.ndarray, h: int, w: int) -> np.ndarray:
        extended = mask.copy()
        bbox = self._get_bbox((mask > 0.5).astype(np.float32))
        if bbox is None:
            return mask

        shoulder_top = bbox['y'] + int(bbox['h'] * 0.18)
        shoulder_bottom = min(bbox['y'] + int(bbox['h'] * 0.68), h)

        for row in range(shoulder_top, shoulder_bottom):
            nonzero = np.where(mask[row, :] > 0.5)[0]
            if len(nonzero) > 0:
                left, right = nonzero[0], nonzero[-1]
                for i in range(self.right_extend):
                    if right + i < w:
                        extended[row, right + i] = max(0, 1.0 - i * 0.05)
                for i in range(self.left_extend):
                    if left - i >= 0:
                        extended[row, left - i] = max(0, 1.0 - i * 0.05)
        return extended

    def _create_mask(self, raw_mask: np.ndarray, h: int, w: int) -> np.ndarray:
        binary = (raw_mask > self.edge_threshold).astype(np.float32)
        filtered = self._filter_false_positives(binary, raw_mask, h, w)

        if filtered.sum() == 0:
            return np.zeros((h, w), dtype=np.float32)

        bbox = self._get_bbox(filtered)
        if bbox is None:
            return np.zeros((h, w), dtype=np.float32)

        head_end = bbox['y'] + int(bbox['h'] * 0.28)
        shoulder_end = bbox['y'] + int(bbox['h'] * 0.52)

        mask_h = np.zeros_like(filtered)
        mask_s = np.zeros_like(filtered)
        mask_b = np.zeros_like(filtered)

        mask_h[:head_end, :] = filtered[:head_end, :]
        mask_s[head_end:shoulder_end, :] = filtered[head_end:shoulder_end, :]
        mask_b[shoulder_end:, :] = filtered[shoulder_end:, :]

        head = cv2.dilate(mask_h, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.head_dilate))
        shoulder = cv2.dilate(mask_s, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.shoulder_dilate))
        body = cv2.dilate(mask_b, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.body_dilate))

        combined = np.maximum.reduce([head, shoulder, body])
        combined = self._extend_shoulders(combined, h, w)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE,
                                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
        return combined

    def _calculate_quality(self, raw_mask: np.ndarray, mask: np.ndarray) -> float:
        binary = raw_mask > self.base_threshold
        if binary.sum() == 0:
            return 0.0

        coverage = mask.sum() / mask.size
        confidence = raw_mask[binary].mean()

        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0

        largest = max(cv2.contourArea(c) for c in contours)
        total = sum(cv2.contourArea(c) for c in contours)
        contiguity = largest / total if total > 0 else 0

        return 0.35 * confidence + 0.35 * min(coverage / 0.25, 1.0) + 0.30 * contiguity

    # ============================================
    # ALPHA MATTING
    # ============================================

    def _create_alpha(self, mask: np.ndarray, raw_mask: np.ndarray,
                      frame: np.ndarray) -> np.ndarray:
        h, w = mask.shape
        alpha = mask.copy()

        # Guided filter
        guide = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        alpha_8 = (alpha * 255).astype(np.uint8)
        alpha = cv2.ximgproc.guidedFilter(guide, alpha_8,
                                          radius=self.preset.edge_refinement,
                                          eps=100).astype(np.float32) / 255.0

        # Joint bilateral
        alpha_8 = (alpha * 255).astype(np.uint8)
        alpha = cv2.ximgproc.jointBilateralFilter(
            frame, alpha_8, d=self.preset.edge_refinement,
            sigmaColor=20, sigmaSpace=20
        ).astype(np.float32) / 255.0

        # Body protection (only within filtered mask)
        alpha[(raw_mask > 0.5) & (mask > 0.3)] = 1.0
        alpha[(raw_mask > 0.25) & (mask > 0.3)] = np.maximum(
            alpha[(raw_mask > 0.25) & (mask > 0.3)], 0.98)
        alpha[(raw_mask > self.protection_threshold) & (mask > 0.3)] = np.maximum(
            alpha[(raw_mask > self.protection_threshold) & (mask > 0.3)], 0.92)
        alpha[mask > 0.5] = np.maximum(alpha[mask > 0.5], 0.88)

        # Feather edges
        edge_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        outer = cv2.dilate(mask, edge_k) - mask
        alpha = np.where(outer > 0, alpha * 0.7 + 0.3 * outer, alpha)

        return np.clip(alpha, 0, 1)

    # ============================================
    # BACKGROUND (BLUR OR REPLACE)
    # ============================================

    def _create_background(self, frame: np.ndarray, h: int, w: int) -> np.ndarray:
        if self.mode == BlurMode.REPLACE and self.background_image is not None:
            return cv2.resize(self.background_image, (w, h))

        # DSLR-style bokeh blur
        radius = self.preset.blur_radius
        size = radius * 2 + 1
        kernel = np.zeros((size, size), np.uint8)
        cv2.circle(kernel, (radius, radius), radius, 1, -1)
        kernel = kernel.astype(np.float32) / kernel.sum()

        bokeh = cv2.filter2D(frame, -1, kernel)
        return cv2.GaussianBlur(bokeh, (35, 35), 0)

    # ============================================
    # OPTICAL FLOW
    # ============================================

    def _detect_motion(self, frame: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_frame_gray is None:
            self.prev_frame_gray = gray
            return True, None

        diff = cv2.absdiff(gray, self.prev_frame_gray)
        has_motion = np.mean(diff) / 255.0 > self.motion_threshold

        flow = None
        if has_motion and self.prev_mask is not None:
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_frame_gray, gray, None,
                0.5, 3, 15, 3, 5, 1.2, 0
            )

        self.prev_frame_gray = gray
        return has_motion, flow

    def _warp_mask(self, mask: np.ndarray, flow: np.ndarray) -> np.ndarray:
        h, w = mask.shape
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        new_x = (x + flow[:, :, 0]).astype(np.float32)
        new_y = (y + flow[:, :, 1]).astype(np.float32)
        return cv2.remap(mask, new_x, new_y, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REPLICATE)

    # ============================================
    # TWO-PASS PROCESSING
    # ============================================

    def _first_pass(self, input_path: str, segmenter) -> Tuple[int, int]:
        cap = cv2.VideoCapture(input_path)
        w, h = int(cap.get(3)), int(cap.get(4))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"\n{'='*60}")
        print(f"PASS 1: Analyzing {total} frames...")
        print(f"{'='*60}")

        self.accumulated_mask = np.zeros((h, w), dtype=np.float32)

        for _ in tqdm(range(total), desc="Analyzing"):
            ret, frame = cap.read()
            if not ret:
                break

            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            raw = segmenter.segment(mp_img).confidence_masks[0].numpy_view()
            raw = cv2.resize(raw, (w, h))

            mask = self._create_mask(raw, h, w)
            quality = self._calculate_quality(raw, mask)

            self.accumulated_mask = np.maximum(self.accumulated_mask, mask)

            if quality > self.best_quality:
                self.best_quality = quality
                self.best_mask = mask.copy()
                self.best_frame_idx = self.frame_count

            self.frame_count += 1

        cap.release()
        self.frame_count = 0  # Reset for second pass

        if self.best_mask is not None:
            self.best_mask = np.maximum(self.best_mask, self.accumulated_mask * 0.85)
            self.best_mask = cv2.morphologyEx(
                self.best_mask, cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
            )

        print(f"\nBest frame: {self.best_frame_idx} (quality={self.best_quality:.3f})")
        return w, h

    def _second_pass(self, input_path: str, output_path: str,
                     segmenter, w: int, h: int, diagnose: bool):
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

        print(f"\n{'='*60}")
        print(f"PASS 2: Rendering ({self.preset.name} quality)...")
        print(f"{'='*60}\n")

        for frame_idx in tqdm(range(total), desc="Rendering"):
            ret, frame = cap.read()
            if not ret:
                break

            start = time.perf_counter()

            # Motion detection
            has_motion, flow = self._detect_motion(frame)

            # Current segmentation
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            raw = segmenter.segment(mp_img).confidence_masks[0].numpy_view()
            raw = cv2.resize(raw, (w, h))

            # Use best mask with motion adaptation
            current = self.best_mask.copy()
            if has_motion and flow is not None and self.prev_mask is not None:
                warped = self._warp_mask(self.prev_mask, flow)
                current = np.maximum(current, warped * 0.9)

            # Temporal smoothing
            self.mask_history.append(current)
            stable = np.maximum.reduce(list(self.mask_history)) if len(self.mask_history) >= 2 else current
            self.prev_mask = stable.copy()

            # Alpha and composite
            alpha = self._create_alpha(stable, raw, frame)
            bg = self._create_background(frame, h, w)

            alpha_3 = alpha[:, :, np.newaxis]
            final = (frame.astype(np.float32) * alpha_3 +
                    bg.astype(np.float32) * (1.0 - alpha_3))

            self.frame_times.append(time.perf_counter() - start)

            if diagnose and frame_idx < 3:
                self._save_diagnostics(frame_idx, frame, raw, stable, alpha, final)

            out.write(final.astype(np.uint8))
            self.frame_count += 1

        cap.release()
        out.release()

    def _save_diagnostics(self, idx: int, frame: np.ndarray, raw: np.ndarray,
                          mask: np.ndarray, alpha: np.ndarray, final: np.ndarray):
        cv2.imwrite(f'diagnostics_pics/diag_f{idx}_input.png', frame)
        cv2.imwrite(f'diagnostics_pics/diag_f{idx}_raw.png', (raw * 255).astype(np.uint8))
        cv2.imwrite(f'diagnostics_pics/diag_f{idx}_mask.png', (mask * 255).astype(np.uint8))
        cv2.imwrite(f'diagnostics_pics/diag_f{idx}_alpha.png', (alpha * 255).astype(np.uint8))
        cv2.imwrite(f'diagnostics_pics/diag_f{idx}_output.png', final.astype(np.uint8))

        # Overlay
        overlay = frame.copy()
        edge = cv2.Canny((alpha * 255).astype(np.uint8), 100, 200)
        overlay[edge > 0] = [0, 255, 0]
        cv2.imwrite(f'diagnostics_pics/diag_f{idx}_overlay.png', overlay)

    # ============================================
    # MAIN ENTRY
    # ============================================

    def process(self, input_path: str, output_path: str, diagnose: bool = False):
        print(f"\n{'='*60}")
        print(f"ZOOM-STYLE BOKEH PROCESSOR")
        print(f"{'='*60}")
        print(f"Preset: {self.preset.name} - {self.preset.description}")
        print(f"Mode: {self.mode.value}")
        print(f"{'='*60}")

        with vision.ImageSegmenter.create_from_options(self.seg_options) as segmenter:
            w, h = self._first_pass(input_path, segmenter)

            if self.best_mask is None:
                print("ERROR: No valid segmentation found!")
                return

            if diagnose:
                cv2.imwrite('diagnostics_pics/diag_best_mask.png', (self.best_mask * 255).astype(np.uint8))

            self._second_pass(input_path, output_path, segmenter, w, h, diagnose)

        # Summary
        avg_ms = np.mean(self.frame_times) * 1000
        fps = 1.0 / np.mean(self.frame_times)

        print(f"\n{'='*60}")
        print("COMPLETE")
        print(f"{'='*60}")
        print(f"Frames: {self.frame_count}")
        print(f"Speed: {avg_ms:.1f}ms/frame ({fps:.1f} FPS)")
        print(f"Output: {output_path}")
        print(f"{'='*60}\n")


# ============================================
# SIMPLE API (Like Zoom's UI)
# ============================================

def blur_background(input_video: str, output_video: str,
                    intensity: str = "high", diagnose: bool = False):
    """
    Simple API matching Zoom's blur background feature.

    Args:
        input_video: Path to input video
        output_video: Path to output video
        intensity: "low", "medium", or "high"
        diagnose: Save diagnostic images
    """
    processor = ZoomStyleBokeh(preset=intensity)
    processor.process(input_video, output_video, diagnose)


def replace_background(input_video: str, output_video: str,
                       background_image: str, diagnose: bool = False):
    """
    Simple API matching Zoom's virtual background feature.

    Args:
        input_video: Path to input video
        output_video: Path to output video
        background_image: Path to background image
        diagnose: Save diagnostic images
    """
    processor = ZoomStyleBokeh(
        preset="high",
        mode=BlurMode.REPLACE,
        background_image=background_image
    )
    processor.process(input_video, output_video, diagnose)


# ============================================
# RUN
# ============================================

if __name__ == "__main__":
    # Option 1: Blur background (like Zoom)
    blur_background('3_seconds_video.mp4', 'ZOOM_BLUR.mp4',
                    intensity="high", diagnose=True)