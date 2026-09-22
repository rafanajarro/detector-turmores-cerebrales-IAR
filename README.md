# Detector de Tumores Cerebrales

Sistema CAD (diagnóstico asistido por computadora) para la **clasificación de imágenes de resonancia magnética (MRI) cerebral** en cuatro categorías: glioma, meningioma, tumor pituitario y sin tumor.

Proyecto de Cátedra **IAR105** — se comparan tres arquitecturas de redes neuronales convolucionales y se expone la mejor a través de una aplicación web en Streamlit.

---

## Resultados

Evaluación sobre el conjunto de prueba (2,414 imágenes no vistas durante el entrenamiento):

| Modelo | Accuracy | Loss | Parámetros | Enfoque |
|--------|:---:|:---:|:---:|---|
| **CNN tradicional** | **95.86 %** | 0.19 | 3.45 M | Entrenada desde cero |
| ResNet50 | 93.45 % | 0.33 | 23.5 M | Transfer learning |
| EfficientNetB0 | 89.77 % | 0.31 | 4.05 M | Transfer learning |

---

## Estructura del repositorio

```
detector-turmores-cerebrales-IAR/
├── app.py                          # Aplicación web (Streamlit)
├── requirements.txt                # Dependencias de la aplicación
├── .streamlit/
│   └── config.toml                 # Tema de la interfaz
├── notebooks/
│   ├── 01_eda.ipynb                # Análisis exploratorio de datos
│   ├── modelo_cnn.ipynb            # Entrenamiento de la CNN
│   ├── modelo_resnet.ipynb         # Entrenamiento de ResNet50
│   └── modelo_efficientnet.ipynb   # Entrenamiento de EfficientNetB0
└── modelo/                         # Modelos entrenados
    ├── cnn.keras
    ├── resnet.keras
    └── efficientnet.keras
```

---

## Requisitos

- **Python 3.12**
- Sistema operativo Windows, macOS o Linux
- ~1 GB de espacio libre para las dependencias

---

## Instalación

1. Clonar el repositorio y entrar en la carpeta:
   ```bash
   git clone https://github.com/rafanajarro/detector-turmores-cerebrales-IAR.git
   cd detector-turmores-cerebrales-IAR
   ```

2. Crear y activar un entorno virtual:
   ```bash
   # Windows (PowerShell)
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1

   # Windows (CMD)
   .venv\Scripts\activate.bat

   # macOS / Linux
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Instalar las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

---

## Modelos entrenados

1. Descargar los modelos: **[modelos.zip](https://github.com/rafanajarro/detector-turmores-cerebrales-IAR/releases/download/modelos-v1/modelos.zip)**
2. Descomprimir y colocar los tres archivos dentro de la carpeta `modelo/`:
   ```
   modelo/cnn.keras
   modelo/resnet.keras
   modelo/efficientnet.keras
   ```

La aplicación muestra únicamente los modelos cuyo archivo esté presente, así que puede ejecutarse con uno solo (por ejemplo, solo `cnn.keras`).

---

## Ejecución de la aplicación

Con el entorno virtual activado:

```bash
streamlit run app.py
```

Se abrirá en el navegador (por defecto en `http://localhost:8501`). En la app se sube una imagen MRI y cada modelo, en su pestaña, muestra la clase predicha, el nivel de confianza y la probabilidad por categoría.

---

## Notebooks

Los notebooks de entrenamiento se ejecutaron en **Google Colab con GPU T4**. Comparten las secciones de preparación (setup, EDA y preprocesamiento) y solo difieren en la definición del modelo.

- `01_eda.ipynb` — caracterización del dataset (distribución de clases, tamaños, canales, ejemplos).
- `modelo_cnn.ipynb` / `modelo_resnet.ipynb` / `modelo_efficientnet.ipynb` — preprocesamiento, entrenamiento, evaluación (accuracy, precision, recall, F1 y matriz de confusión) y curvas.

---

## Dataset

**Brain Tumor MRI Dataset** (Mendeley Data) — 12,064 imágenes T1 con contraste, divididas en `Training/` y `Testing/`, cada una con las cuatro clases. No se incluye en el repositorio.

- Fuente oficial: [Brain Tumor MRI Dataset — Mendeley Data](https://data.mendeley.com/datasets/zwr4ntf94j/1)

---
