import os
import urllib.request

# Diccionario con nombre del archivo y su URL
MODELS = {
    "body_segmenter.tflite": "https://storage.googleapis.com/mediapipe-models/image_segmenter/body_segmenter/float16/latest/body_segmenter.tflite",
    "selfie_segmenter.tflite": "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite",
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
    "pose_landmarker.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
}

def download_models():
    # Creamos una carpeta 'models' para mantener ordenado el proyecto
    if not os.path.exists('models'):
        os.makedirs('models')
        print("Carpeta 'models' creada.")

    for filename, url in MODELS.items():
        path = os.path.join('models', filename)
        
        if os.path.exists(path):
            print(f"✅ {filename} ya existe, saltando descarga.")
        else:
            print(f"⏳ Descargando {filename}...")
            try:
                urllib.request.urlretrieve(url, path)
                print(f"    Descarga completa: {path}")
            except Exception as e:
                print(f"❌ Error descargando {filename}: {e}")

if __name__ == "__main__":
    print("Iniciando configuración de modelos MediaPipe...")
    download_models()
    print("\nTodo listo para trabajar.")