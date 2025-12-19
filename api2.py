import os
import logging
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from pydantic import BaseModel
from typing import List, Dict, Optional
import tempfile
from pathlib import Path

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurar semillas para reproducibilidad
tf.random.set_seed(42)
np.random.seed(42)

# Inicializar FastAPI
app = FastAPI(
    title="API de Detección de Enfermedades en Plantas",
    description="API para detectar enfermedades en hojas de tomate, papa y pimiento",
    version="1.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Diccionario de etiquetas en español
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
    """Convierte nombre de carpeta en etiqueta en español"""
    if folder_name in ES_LABELS:
        return ES_LABELS[folder_name]
    return folder_name.replace("___", " – ").replace("__", " ").replace("_", " ")

# Variables globales
modelo = None
class_names = None
image_shape = (256, 256)

# Modelos Pydantic para respuestas
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

@app.on_event("startup")
async def cargar_modelo():
    """Carga el modelo y las clases al iniciar la aplicación"""
    global modelo, class_names
    
    try:
        BASE_DIR = Path(__file__).resolve().parent
        MODEL_PATH = BASE_DIR / "modelo_multiple_enfermedades_hojas_de_plantas.keras"
        
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"No se encontró el modelo en: {MODEL_PATH}")
        
        logger.info(f"Cargando modelo desde: {MODEL_PATH}")
        modelo = load_model(MODEL_PATH)
        logger.info("✅ Modelo cargado exitosamente")
        
        # Obtener orden de clases desde el dataset
        train_path = "dataset/train"
        
        if os.path.exists(train_path):
            temp_datagen = ImageDataGenerator(rescale=1/255)
            temp_generator = temp_datagen.flow_from_directory(
                train_path,
                target_size=image_shape,
                batch_size=1,
                class_mode="categorical",
                shuffle=False
            )
            class_names = list(temp_generator.class_indices.keys())
            logger.info(f"✅ Clases cargadas: {len(class_names)} categorías")
        else:
            # Fallback: usar orden basado en ES_LABELS
            class_names = list(ES_LABELS.keys())
            logger.warning(f"⚠️ No se encontró dataset, usando orden predeterminado de clases")
            
    except Exception as e:
        logger.error(f"❌ Error al cargar el modelo: {str(e)}")
        raise

def predecir_imagen(
    ruta_imagen: str,
    umbral_confianza: float = 0.70,
    umbral_entropia: float = 0.75
) -> Dict:
    """
    Predice con detección avanzada de imágenes fuera de contexto.
    """
    if not os.path.exists(ruta_imagen):
        raise FileNotFoundError(f"No se encontró la imagen: {ruta_imagen}")
    
    # Cargar y preprocesar
    img = load_img(ruta_imagen, target_size=image_shape)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predicción
    prediccion = modelo.predict(img_array, verbose=0)
    clase_pred_idx = np.argmax(prediccion[0])
    confianza = float(prediccion[0][clase_pred_idx])
    
    # Nombres
    clase_pred_nombre = class_names[clase_pred_idx]
    clase_pred_es = es_label(clase_pred_nombre)
    
    # Top 3
    top3_idx = np.argsort(prediccion[0])[-3:][::-1]
    top3 = [
        {
            "nombre": class_names[i],
            "nombre_es": es_label(class_names[i]),
            "probabilidad": float(prediccion[0][i])
        }
        for i in top3_idx
    ]
    
    # CRITERIOS DE VALIDACIÓN
    criterio_confianza = confianza >= umbral_confianza
    
    # Entropía
    entropia = -np.sum(prediccion[0] * np.log(prediccion[0] + 1e-10))
    entropia_normalizada = float(entropia / np.log(len(class_names)))
    criterio_entropia = entropia_normalizada <= umbral_entropia
    
    # Gap entre top 1 y top 2
    gap_top1_top2 = float(prediccion[0][top3_idx[0]] - prediccion[0][top3_idx[1]])
    criterio_gap = gap_top1_top2 >= 0.20
    
    # Suma de top 3
    suma_top3 = float(sum([prediccion[0][i] for i in top3_idx]))
    criterio_concentracion = suma_top3 >= 0.80
    
    # Decisión final
    es_valida = criterio_confianza and criterio_entropia
    nivel_certeza = "ALTA" if (criterio_confianza and criterio_entropia and criterio_gap) else \
                    "MEDIA" if criterio_confianza else "BAJA"
    
    # Mensajes y razones
    mensaje = None
    razones_rechazo = None
    
    if es_valida:
        mensaje = f"Predicción válida con certeza {nivel_certeza}"
    else:
        mensaje = "Imagen rechazada - No es válida para este modelo"
        razones_rechazo = []
        if not criterio_confianza:
            razones_rechazo.append(f"Confianza muy baja ({confianza*100:.2f}% < {umbral_confianza*100:.0f}%)")
        if not criterio_entropia:
            razones_rechazo.append(f"Alta incertidumbre (entropía: {entropia_normalizada*100:.2f}%)")
        razones_rechazo.append("La imagen puede no ser una hoja de tomate, papa o pimiento")
    
    return {
        'clase_predicha': clase_pred_nombre if es_valida else None,
        'clase_predicha_es': clase_pred_es if es_valida else None,
        'confianza': confianza,
        'es_valida': es_valida,
        'nivel_certeza': nivel_certeza,
        'entropia': entropia_normalizada,
        'gap_top1_top2': gap_top1_top2,
        'metricas': {
            'confianza_principal': confianza,
            'incertidumbre_entropia': entropia_normalizada,
            'gap_top1_top2': gap_top1_top2,
            'concentracion_top3': suma_top3
        },
        'criterios': {
            'confianza': criterio_confianza,
            'entropia': criterio_entropia,
            'gap': criterio_gap,
            'concentracion': criterio_concentracion
        },
        'top3': top3,
        'mensaje': mensaje,
        'razones_rechazo': razones_rechazo
    }

@app.post("/predecir", response_model=ResultadoPrediccion)
async def predecir(
    file: UploadFile = File(...),
    umbral_confianza: float = 0.70,
    umbral_entropia: float = 0.75
):
    """
    Endpoint para predecir enfermedades en hojas de plantas.
    
    Args:
        file: Archivo de imagen (JPG, JPEG, PNG)
        umbral_confianza: Umbral mínimo de confianza (default: 0.70)
        umbral_entropia: Umbral máximo de entropía (default: 0.75)
    
    Returns:
        Resultado de la predicción con métricas detalladas
    """
    
    # Validar tipo de archivo
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="El archivo debe ser una imagen (JPG, JPEG, PNG)"
        )
    
    # Guardar archivo temporalmente
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        # Realizar predicción
        resultado = predecir_imagen(
            temp_path,
            umbral_confianza=umbral_confianza,
            umbral_entropia=umbral_entropia
        )
        
        logger.info(f"✅ Predicción exitosa: {resultado.get('clase_predicha_es', 'No válida')}")
        
        return resultado
        
    except Exception as e:
        logger.error(f"❌ Error en predicción: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error al procesar la imagen: {str(e)}")
    
    finally:
        # Limpiar archivo temporal
        if os.path.exists(temp_path):
            os.unlink(temp_path)

@app.get("/")
async def root():
    """Endpoint raíz con información de la API"""
    return {
        "mensaje": "API de Detección de Enfermedades en Plantas",
        "version": "1.0.0",
        "endpoints": {
            "prediccion": "/predecir",
            "documentacion": "/docs",
            "clases": "/clases"
        }
    }

@app.get("/clases")
async def obtener_clases():
    """Retorna todas las clases disponibles"""
    if class_names is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    
    return {
        "total": len(class_names),
        "clases": [
            {
                "indice": i,
                "nombre": cls,
                "nombre_es": es_label(cls)
            }
            for i, cls in enumerate(class_names)
        ]
    }

@app.get("/health")
async def health_check():
    """Verifica el estado de la API"""
    return {
        "status": "healthy",
        "modelo_cargado": modelo is not None,
        "clases_cargadas": class_names is not None,
        "total_clases": len(class_names) if class_names else 0
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)