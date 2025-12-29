🎥 # Zoom-Style Bokeh Processor (IA & Computer Vision)
"Lo que comenzó como una pregunta curiosa mientras veía un tutorial y comía palomitas, se convirtió en un motor de procesamiento de video completo..."Este repositorio documenta mi camino para entender y replicar los filtros de segmentación de videollamadas (como Zoom o Teams). Es la prueba de que un problema cotidiano es la mejor oportunidad para aplicar ingeniería de datos y visión artificial de alto nivel.

🚀 # Resumen del ProyectoDesarrollé un pipeline de procesamiento de video offline que utiliza Inteligencia Artificial para segmentar personas y aplicar efectos de desenfoque (Bokeh) o fondos virtuales. El enfoque principal fue la fidelidad visual y la estabilidad temporal, superando las limitaciones comunes de los filtros en tiempo real.

🧠 # Características Técnicas (ML & CV)Para elevar este script a un estándar de Data Science, se implementaron:Segmentación Semántica: Integración de modelos ligeros TFLite (MediaPipe) optimizados con delegados XNNPACK para CPU.Estabilidad Temporal: Implementación de Flujo Óptico (Farneback) para que la máscara "siga" el movimiento del usuario, evitando el parpadeo visual.Refinamiento de Bordes: Uso de Guided Filters y Bilateral Filters para procesar el canal alfa, logrando una integración natural en áreas complejas como el cabello.Arquitectura de Doble Pasada: Una fase de análisis para identificar el "Mejor Frame" de referencia y una fase de renderizado compuesto.

📈 # Análisis de Rendimiento (Benchmarking)Como Data Scientist, el monitoreo es clave. Estos son los resultados en una prueba con un video de 115 cuadros (3 segundos):MétricaResultadoTiempo por Cuadro259.0 msFPS de Procesamiento3.9 FPSLatencia de InferenciaOptimizado con delegados XNNPACKNota: Se priorizó la precisión (Preset: High) sobre la velocidad, logrando una calidad superior a las implementaciones estándar de tiempo real.

🛠️ # Cómo usarloEl script ofrece una API sencilla, ocultando la complejidad del motor interno:Pythonfrom zoom_like_sc import blur_background

# Aplicar desenfoque profesional
blur_background(
    input_video='mi_video.mp4', 
    output_video='resultado.mp4', 
    intensity="high"
)
🎓 # Aprendizajes ClaveOptimización de Modelos: Cómo trabajar con modelos cuantizados para mejorar la velocidad en CPU.Composición Digital: El uso de máscaras de confianza como canales alfa.Ingeniería de Software: Aplicación de patrones de diseño (Encapsulamiento y Dataclasses) para crear herramientas mantenibles.
