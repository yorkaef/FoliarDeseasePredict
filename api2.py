import os
import logging
import numpy as np
import asyncio
import tempfile
import shutil
import urllib.request
from pathlib import Path
from typing import List, Dict, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

tf.random.set_seed(42)
np.random.seed(42)

app = FastAPI(
    title="API de Detección de Enfermedades en Plantas",
    description="API para detectar enfermedades en hojas de tomate, papa y pimiento",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ES_LABELS = {
    "Pepper__bell___Bacterial_spot": "Pimiento morrón – Mancha bacteriana",
    "Pepper__bell___healthy": "Pimiento morrón – Sano",
    "Potato___Early_blight": "Papa – Tizón temprano",
    "Potato___healthy": "Papa – Sano",
    "Potato___Late_blight": "Papa – Tizón tardío",
    "Tomato__Target_Spot": "Tomate – Mancha diana",
    "Tomato__Tomato_mosaic_virus": "Tomate – Virus del mosaico (ToMV)",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "Tomate – Virus del rizado amarillo (TYLCV)",
    "Tomato_Bacterial_spot": "Tomate – Mancha bacteriana",
    "Tomato_Early_blight": "Tomate – Tizón temprano",
    "Tomato_healthy": "Tomate – Sano",
    "Tomato_Late_blight": "Tomate – Tizón tardío",
    "Tomato_Leaf_Mold": "Tomate – Moho de la hoja",
    "Tomato_Septoria_leaf_spot": "Tomate – Mancha foliar por Septoria",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Tomate – Ácaros (araña de dos manchas)",
}

def es_label(folder_name: str) -> str:
    return ES_LABELS.get(folder_name, folder_name.replace("___", " – ").replace("__", " ").replace("_", " "))

# Paths / estado global
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "modelo_multiple_enfermedades_hojas_de_plantas.keras"

modelo = None
class_names = None
image_shape = (256, 256)

model_ready = asyncio.Event()
model_error: Optional[str] = None

# Schemas
class PrediccionTop(BaseModel):
    nombre: str
    nombre_es: str
    probabilidad: float

class Criterios(BaseModel):
    confianza: bool
    entropia: bool
    gap: bool
    concentracion: bool

class Metricas(BaseModel):
    confianza_principal: float
    incertidumbre_entropia: float
    gap_top1_top2: float
    concentracion_top3: float

class ResultadoPrediccion(BaseModel):
    clase_predicha: Optional[str]
    clase_predicha_es: Optional[str]
    confianza: float
    es_valida: bool
    nivel_certeza: str
    entropia: float
    gap_top1_top2: float
    metricas: Metricas
    criterios: Criterios
    top3: List[PrediccionTop]
    mensaje: Optional[str] = None
    razones_rechazo: Optional[List[str]] = None

def _archivo_es_lfs_pointer(path: Path) -> bool:
    if not path.exists():
        return False
    if path.stat().st_size > 5_000_000:  # si ya pesa bastante, no es puntero
        return False
    head = path.read_bytes()[:200]
    return b"git-lfs" in head or b"oid sha256" in head

def _descargar_modelo(url: str, dest: Path) -> None:
    logger.info(f"⬇️ Descargando modelo desde URL a: {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)
    logger.info(f"✅ Modelo descargado. Tamaño: {dest.stat().st_size/1024/1024:.2f} MB")

async def _cargar_modelo_bg():
    global modelo, class_names, model_error
    try:
        url = os.getenv("MODEL_URL")

        # si falta el archivo o es puntero LFS, descargamos
        if (not MODEL_PATH.exists()) or _archivo_es_lfs_pointer(MODEL_PATH):
            if not url:
                raise RuntimeError("Falta MODEL_URL en variables de entorno para descargar el modelo.")
            await asyncio.to_thread(_descargar_modelo, url, MODEL_PATH)

        logger.info(f"📦 Cargando modelo desde: {MODEL_PATH}")
        modelo_local = await asyncio.to_thread(load_model, str(MODEL_PATH))
        modelo = modelo_local
        logger.info("✅ Modelo cargado exitosamente")

        # Clases
        train_path = BASE_DIR / "dataset" / "train"
        if train_path.exists():
            def _leer_clases():
                temp_datagen = ImageDataGenerator(rescale=1/255)
                temp_generator = temp_datagen.flow_from_directory(
                    str(train_path),
                    target_size=image_shape,
                    batch_size=1,
                    class_mode="categorical",
                    shuffle=False
                )
                return list(temp_generator.class_indices.keys())

            class_names = await asyncio.to_thread(_leer_clases)
        else:
            class_names = list(ES_LABELS.keys())
            logger.warning("⚠️ No se encontró dataset/train, usando orden predeterminado de clases")

    except Exception as e:
        model_error = str(e)
        logger.error(f"❌ Error al cargar el modelo: {model_error}")
    finally:
        model_ready.set()

@app.on_event("startup")
async def startup():
    # No bloquea el arranque
    asyncio.create_task(_cargar_modelo_bg())

async def asegurar_modelo():
    await model_ready.wait()
    if modelo is None or class_names is None:
        raise HTTPException(status_code=503, detail=f"Modelo no disponible: {model_error}")

def predecir_imagen(ruta_imagen: str, umbral_confianza: float = 0.70, umbral_entropia: float = 0.75) -> Dict:
    if not os.path.exists(ruta_imagen):
        raise FileNotFoundError(f"No se encontró la imagen: {ruta_imagen}")

    img = load_img(ruta_imagen, target_size=image_shape)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediccion = modelo.predict(img_array, verbose=0)
    clase_pred_idx = int(np.argmax(prediccion[0]))
    confianza = float(prediccion[0][clase_pred_idx])

    clase_pred_nombre = class_names[clase_pred_idx]
    clase_pred_es = es_label(clase_pred_nombre)

    top3_idx = np.argsort(prediccion[0])[-3:][::-1]
    top3 = [{
        "nombre": class_names[i],
        "nombre_es": es_label(class_names[i]),
        "probabilidad": float(prediccion[0][i])
    } for i in top3_idx]

    criterio_confianza = confianza >= umbral_confianza

    entropia = -np.sum(prediccion[0] * np.log(prediccion[0] + 1e-10))
    entropia_normalizada = float(entropia / np.log(len(class_names)))
    criterio_entropia = entropia_normalizada <= umbral_entropia

    gap_top1_top2 = float(prediccion[0][top3_idx[0]] - prediccion[0][top3_idx[1]])
    criterio_gap = gap_top1_top2 >= 0.20

    suma_top3 = float(sum([prediccion[0][i] for i in top3_idx]))
    criterio_concentracion = suma_top3 >= 0.80

    es_valida = criterio_confianza and criterio_entropia
    nivel_certeza = "ALTA" if (criterio_confianza and criterio_entropia and criterio_gap) else \
                    "MEDIA" if criterio_confianza else "BAJA"

    if es_valida:
        mensaje = f"Predicción válida con certeza {nivel_certeza}"
        razones_rechazo = None
    else:
        mensaje = "Imagen rechazada - No es válida para este modelo"
        razones_rechazo = []
        if not criterio_confianza:
            razones_rechazo.append(f"Confianza muy baja ({confianza*100:.2f}% < {umbral_confianza*100:.0f}%)")
        if not criterio_entropia:
            razones_rechazo.append(f"Alta incertidumbre (entropía: {entropia_normalizada*100:.2f}%)")
        razones_rechazo.append("La imagen puede no ser una hoja de tomate, papa o pimiento")

    return {
        "clase_predicha": clase_pred_nombre if es_valida else None,
        "clase_predicha_es": clase_pred_es if es_valida else None,
        "confianza": confianza,
        "es_valida": es_valida,
        "nivel_certeza": nivel_certeza,
        "entropia": entropia_normalizada,
        "gap_top1_top2": gap_top1_top2,
        "metricas": {
            "confianza_principal": confianza,
            "incertidumbre_entropia": entropia_normalizada,
            "gap_top1_top2": gap_top1_top2,
            "concentracion_top3": suma_top3
        },
        "criterios": {
            "confianza": criterio_confianza,
            "entropia": criterio_entropia,
            "gap": criterio_gap,
            "concentracion": criterio_concentracion
        },
        "top3": top3,
        "mensaje": mensaje,
        "razones_rechazo": razones_rechazo
    }

@app.post("/predecir", response_model=ResultadoPrediccion)
async def predecir(
    file: UploadFile = File(...),
    umbral_confianza: float = 0.70,
    umbral_entropia: float = 0.75
):
    """Endpoint para predecir enfermedades en hojas de plantas."""
    await asegurar_modelo()

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen (JPG, JPEG, PNG)")

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name

        return predecir_imagen(temp_path, umbral_confianza=umbral_confianza, umbral_entropia=umbral_entropia)

    except Exception as e:
        logger.error(f"❌ Error en predicción: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error al procesar la imagen: {str(e)}")

    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)

@app.get("/")
async def root():
    return {"mensaje": "API OK", "docs": "/docs", "health": "/health"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "modelo_cargando": not model_ready.is_set(),
        "modelo_cargado": modelo is not None,
        "error": model_error
    }
