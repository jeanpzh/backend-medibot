from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import os
import json
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Cargar variables de entorno desde el archivo .env
load_dotenv()

# --- Pydantic Model for Input Data ---
class CovidFeatures(BaseModel):
    breathing_problem: int
    fever: int
    dry_cough: int
    sore_throat: int
    hyper_tension: int
    abroad_travel: int
    contact_with_covid_patient: int
    attended_large_gathering: int
    visited_public_exposed_places: int
    family_working_in_public_exposed_places: int

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Carga el modelo de ML al inicio de la aplicación.
    """
    global model_pipeline
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Archivo del modelo no encontrado! Asegúrate de que 'covid_severity_model_pipeline.joblib' exista en el directorio 'backend'.")
    model_pipeline = joblib.load(MODEL_PATH)
    yield
    # Clean up the model if needed (e.g. model_pipeline = None)

# --- FastAPI App Initialization ---
app = FastAPI(
    title="API de Predicción de COVID-19",
    description="Una API para predecir la probabilidad de tener COVID-19 basada en síntomas y exposición.",
    version="1.0.0",
    lifespan=lifespan
)

# Configuración de CORS para permitir solicitudes desde el frontend
# Se lee la URL del frontend desde una variable de entorno para flexibilidad
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:8000")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# --- Load Model --- , esta en /model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "covid_severity_model_pipeline.joblib")
model_pipeline = None

EXPECTED_FEATURES = [
    'breathing_problem', 'fever', 'dry_cough', 'sore_throat',
    'hyper_tension', 'abroad_travel', 'contact_with_covid_patient',
    'attended_large_gathering', 'visited_public_exposed_places',
    'family_working_in_public_exposed_places'
]

@app.get("/", summary="Verificación de estado", description="Endpoint para verificar si la API está en funcionamiento.")
def read_root():
    return {"status": "ok", "message": "API de Predicción de COVID-19 está en línea."}

@app.post("/predict", summary="Predecir COVID-19", description="Recibe los síntomas y factores de un paciente y predice si tiene COVID-19.")
def predict_covid(features: CovidFeatures):
    """
    Realiza una predicción de COVID-19.

    - **features**: Un objeto JSON con los síntomas y factores de exposición.
    """
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Modelo no está cargado. Revisa los logs del servidor.")

    try:
        # Convert Pydantic model to dictionary
        input_data_dict = features.dict()

        # Build the input with columns in the correct order to match the model's training
        ordered_input_data = {col: input_data_dict.get(col, 0) for col in EXPECTED_FEATURES}
        input_df = pd.DataFrame([ordered_input_data])

        # Make prediction
        prediction = model_pipeline.predict(input_df)
        probability = model_pipeline.predict_proba(input_df)[0][int(prediction[0])]

        return {
            "predicted_covid": int(prediction[0]),
            "label": "COVID-19" if prediction[0] == 1 else "NO COVID-19",
            "confidence": round(float(probability), 4)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ocurrió un error durante la predicción: {e}")

