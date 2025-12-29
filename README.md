# 🎥 Zoom-Style Bokeh Processor (IA & Computer Vision)

> "Lo que comenzó como una pregunta curiosa mientras veía un tutorial y comía palomitas, se convirtió en un motor de procesamiento de video completo..."

Este repositorio documenta mi camino para entender y replicar los filtros de segmentación de videollamadas (como Zoom o Teams). Es la prueba de que un problema cotidiano es la mejor oportunidad para aplicar ingeniería de datos y visión artificial de alto nivel.

---

## 🚀 Resumen del Proyecto
Desarrollé un pipeline de procesamiento de video **offline** que utiliza Inteligencia Artificial para segmentar personas y aplicar efectos de desenfoque (Bokeh) o fondos virtuales. El enfoque principal fue la **fidelidad visual** y la **estabilidad temporal**, superando las limitaciones comunes de los filtros en tiempo real.



## 🧠 Características Técnicas


* **Segmentación Semántica:** Integración de modelos ligeros **TFLite (MediaPipe)** optimizados con delegados XNNPACK para CPU.
* **Estabilidad Temporal:** Implementación de **Flujo Óptico (Farneback)** para que la máscara "siga" el movimiento del usuario, eliminando el parpadeo visual (*flickering*).
* **Refinamiento de Bordes:** Uso de **Guided Filters** y **Bilateral Filters** para procesar el canal alfa, logrando una integración natural en áreas complejas como el cabello.
* **Arquitectura de Doble Pasada:** Una fase de análisis estadístico para identificar el "Mejor Frame" de referencia y una fase de renderizado compuesto de alta precisión.



## 📈 Análisis de Rendimiento (Benchmarking)
Como Data Scientist, el monitoreo de métricas es fundamental. Resultados obtenidos en una prueba con un video de **115 cuadros (3 segundos)**:

| Métrica | Resultado |
| :--- | :--- |
| **Tiempo por Cuadro** | 259.0 ms |
| **FPS de Procesamiento** | 3.9 FPS |
| **Latencia de Inferencia** | Optimizado con delegados XNNPACK |

> **Nota:** Se priorizó la precisión (**Preset: High**) sobre la velocidad, logrando una calidad superior a las implementaciones estándar de tiempo real.

## 🛠️ Cómo usarlo
El script ofrece una API de alto nivel, ocultando la complejidad del motor interno para el usuario final:

```python
from zoom_like_sc import blur_background

# Aplicar desenfoque de profundidad profesional
blur_background(
    input_video='mi_video.mp4', 
    output_video='resultado.mp4', 
    intensity="high"
)
```
## 🎓 Aprendizajes Clave

**Optimización de Modelos:** Trabajo con modelos cuantizados para maximizar la velocidad en arquitecturas CPU.

**Composición Digital:** Manejo de máscaras de confianza (confidence masks) como canales alfa de precisión.

**Ingeniería de Software:** Aplicación de patrones de diseño como Encapsulamiento y DataClasses para construir herramientas escalables y mantenibles.
