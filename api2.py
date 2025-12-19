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

from PIL import Image
from tflite_runtime.interpreter import Interpreter


# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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


# ===== Paths / estado global =====
BASE_DIR = Path(__file__).resolve().parent

# Nombre del archivo local (puedes cambiarlo con env MODEL_FILE)
MODEL_FILE = os.getenv("MODEL_FILE", "model.tflite")
MODEL_PATH = BASE_DIR / MODEL_FILE

# Si usas Git LFS, pon MODEL_URL en Render para descargar el archivo real
MODEL_URL = os.getenv("MODEL_URL", "")

# Clases: puedes definir CLASSES (comma separated) o poner classes.txt
CLASSES_ENV = os.getenv("CLASSES", "")

interpreter: Optional[Interpreter] = None
input_details = None
output_details = None

class_names: Optional[List[str]] = None
image_shape = (256, 256)  # se ajustará según el modelo

model_ready = asyncio.Event()
model_error: Optional[str] = None


# ===== Schemas =====
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
    """Detecta si el archivo es un puntero de Git LFS (chiquito y con texto git-lfs)."""
    if not path.exists():
        return False
    if path.stat().st_size > 5_000_000:
        return False
    head = path.read_bytes()[:300]
    return (b"git-lfs" in head) or (b"oid sha256" in head)

def _descargar_modelo(url: str, dest: Path) -> None:
    logger.info(f"⬇️ Descargando modelo a: {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)
    logger.info(f"✅ Modelo descargado. Tamaño: {dest.stat().st_size/1024/1024:.2f} MB")

def _cargar_clases() -> List[str]:
    if CLASSES_ENV.strip():
        return [x.strip() for x in CLASSES_ENV.split(",") if x.strip()]

    classes_txt = BASE_DIR / "classes.txt"
    if classes_txt.exists():
        lines = [ln.strip() for ln in classes_txt.read_text(encoding="utf-8").splitlines()]
        lines = [ln for ln in lines if ln]
        if lines:
            return lines

    # fallback
    return list(ES_LABELS.keys())


async def _cargar_modelo_bg():
    global interpreter, input_details, output_details, class_names, image_shape, model_error
    try:
        # Si no existe o es puntero LFS: descargar desde MODEL_URL
        if (not MODEL_PATH.exists()) or _archivo_es_lfs_pointer(MODEL_PATH):
            if not MODEL_URL:
                raise RuntimeError(
                    "Falta el modelo. Sube un .tflite al repo o configura MODEL_URL en Render."
                )
            await asyncio.to_thread(_descargar_modelo, MODEL_URL, MODEL_PATH)

        logger.info(f"📦 Cargando TFLite desde: {MODEL_PATH}")
        interpreter_local = Interpreter(model_path=str(MODEL_PATH))
        interpreter_local.allocate_tensors()

        in_details = interpreter_local.get_input_details()
        out_details = interpreter_local.get_output_details()

        # Normalmente: [1, H, W, C]
        in_shape = in_details[0]["shape"]
        if len(in_shape) >= 3:
            h, w = int(in_shape[1]), int(in_shape[2])
            if h > 0 and w > 0:
                image_shape = (h, w)

        interpreter = interpreter_local
        input_details = in_details
        output_details = out_details

        class_names = _cargar_clases()

        # Aviso si no coincide cantidad de clases con salida del modelo
        out_shape = out_details[0]["shape"]
        n_out = int(out_shape[-1]) if len(out_shape) else None
        if n_out and len(class_names) != n_out:
            logger.warning(f"⚠️ Clases ({len(class_names)}) != salida del modelo ({n_out}). "
                           f"Revisa el orden/cantidad de clases.")

        logger.info(f"✅ Modelo listo. Input: {image_shape}, clases: {len(class_names)}")

    except Exception as e:
        model_error = str(e)
        logger.error(f"❌ Error al cargar el modelo: {model_error}")

    finally:
        model_ready.set()


@app.on_event("startup")
async def startup():
    asyncio.create_task(_cargar_modelo_bg())


async def asegurar_modelo():
    await model_ready.wait()
    if interpreter is None or class_names is None:
        raise HTTPException(status_code=503, detail=f"Modelo no disponible: {model_error}")


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-12)


def _preprocesar_imagen(ruta_imagen: str) -> np.ndarray:
    # PIL -> RGB -> resize -> float32 [0,1]
    img = Image.open(ruta_imagen).convert("RGB")
    img = img.resize(image_shape)
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = np.expand_dims(x, axis=0)  # [1,H,W,C]
    return x


def _infer_tflite(x: np.ndarray) -> np.ndarray:
    """Corre inferencia y devuelve vector de probabilidades (float)."""
    in_info = input_details[0]
    out_info = output_details[0]

    x_in = x

    # Ajuste por cuantización de entrada si aplica
    if in_info["dtype"] in (np.uint8, np.int8):
        scale, zero = in_info.get("quantization", (0.0, 0))
        if scale and scale > 0:
            x_in = (x_in / scale + zero).astype(in_info["dtype"])
        else:
            x_in = (x_in * 255.0).astype(in_info["dtype"])
    else:
        x_in = x_in.astype(in_info["dtype"])

    interpreter.set_tensor(in_info["index"], x_in)
    interpreter.invoke()
    y = interpreter.get_tensor(out_info["index"])[0]

    # De-cuantizar salida si aplica
    if out_info["dtype"] in (np.uint8, np.int8):
        scale, zero = out_info.get("quantization", (0.0, 0))
        y = y.astype(np.float32)
        if scale and scale > 0:
            y = (y - zero) * scale

    y = y.astype(np.float32)

    # Si no parece probabilidad, aplicamos softmax
    s = float(np.sum(y))
    if not (0.98 <= s <= 1.02) or np.any(y < 0) or np.any(y > 1.0):
        y = _softmax(y)

    return y


def predecir_imagen(ruta_imagen: str, umbral_confianza: float = 0.70, umbral_entropia: float = 0.75) -> Dict:
    if not os.path.exists(ruta_imagen):
        raise FileNotFoundError(f"No se encontró la imagen: {ruta_imagen}")

    x = _preprocesar_imagen(ruta_imagen)
    pred = _infer_tflite(x)  # vector [n_clases]

    clase_pred_idx = int(np.argmax(pred))
    confianza = float(pred[clase_pred_idx])

    clase_pred_nombre = class_names[clase_pred_idx]
    clase_pred_es = es_label(clase_pred_nombre)

    top3_idx = np.argsort(pred)[-3:][::-1]
    top3 = [{
        "nombre": class_names[i],
        "nombre_es": es_label(class_names[i]),
        "probabilidad": float(pred[i])
    } for i in top3_idx]

    criterio_confianza = confianza >= umbral_confianza

    entropia = -float(np.sum(pred * np.log(pred + 1e-10)))
    entropia_normalizada = float(entropia / np.log(len(class_names)))
    criterio_entropia = entropia_normalizada <= umbral_entropia

    gap_top1_top2 = float(pred[top3_idx[0]] - pred[top3_idx[1]])
    criterio_gap = gap_top1_top2 >= 0.20

    suma_top3 = float(np.sum(pred[top3_idx]))
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
    await asegurar_modelo()

    if not file.content_type or not file.content_type.startswith("image/"):
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
    return {"mensaje": "API OK", "docs": "/docs", "health": "/health", "clases": "/clases"}


@app.get("/clases")
async def clases():
    await asegurar_modelo()
    return {"total": len(class_names), "clases": [{"indice": i, "nombre": c, "nombre_es": es_label(c)} for i, c in enumerate(class_names)]}


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "modelo_cargando": not model_ready.is_set(),
        "modelo_cargado": interpreter is not None,
        "model_file": str(MODEL_PATH),
        "model_exists": MODEL_PATH.exists(),
        "image_shape": list(image_shape),
        "total_clases": len(class_names) if class_names else 0,
        "error": model_error
    }
