from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import os
import json
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import shap
import google.generativeai as genai

# Cargar variables de entorno desde el archivo .env
load_dotenv()

# --- Configuración de la API de Gemini ---
# La clave se cargará automáticamente desde la variable de entorno GOOGLE_API_KEY
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    # Este error se mostrará al iniciar el servidor si la clave no está presente
    raise RuntimeError("La variable de entorno GOOGLE_API_KEY no está configurada.")
genai.configure(api_key=api_key)  # type: ignore

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

class MonitoringEntry(BaseModel):
    entry_date: str
    temperature: float | None = None
    oxygen_level: float | None = None
    heart_rate: int | None = None
    symptoms: list[str] | None = []
    overall_feeling: int | None = None
    notes: str | None = ""

class ExplainRequest(BaseModel):
    features: CovidFeatures
    prediction_label: str = Field(..., description="El resultado de la predicción, ej. 'Positivo para COVID-19'")
    monitoring_data: list[MonitoringEntry] | None = Field(None, description="Historial opcional de datos de monitoreo del usuario.")

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

# Configuración de CORS para permitir solicitudes desde el frontend en desarrollo
origins = [
    "http://localhost:3000"]

# Añadir la URL del frontend desde variables de entorno si existe
FRONTEND_URL = os.getenv("FRONTEND_URL")
if FRONTEND_URL:
    origins.append(FRONTEND_URL)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
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

# --- SHAP Explainer ---
explainer = None

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

@app.post("/explain", summary="Generar explicación estructurada de la predicción", description="Recibe datos y predicción, y devuelve un JSON estructurado para la UI.")
def explain_prediction(request: ExplainRequest):
    """
    Genera una explicación estructurada en formato JSON para la predicción del modelo.

    - **request**: Un objeto JSON con `features` (síntomas), `prediction_label` y opcionalmente `monitoring_data`.
    """
    
    try:
        input_data_dict = request.features.dict()
        prediction_label = request.prediction_label
        monitoring_data = request.monitoring_data

        features_es = {
            'breathing_problem': 'dificultad para respirar', 
            'fever': 'fiebre', 
            'dry_cough': 'tos seca',
            'sore_throat': 'dolor de garganta', 
            'hyper_tension': 'hipertensión', 
            'abroad_travel': 'viaje al extranjero',
            'contact_with_covid_patient': 'contacto con paciente COVID', 
            'attended_large_gathering': 'asistencia a grandes reuniones',
            'visited_public_exposed_places': 'visita a lugares públicos expuestos',
            'family_working_in_public_exposed_places': 'familiares trabajando en lugares públicos'
        }
        
        symptoms_present = []
        symptoms_absent = []
        
        for feature, value in input_data_dict.items():
            feature_name = features_es.get(feature, feature)
            if value == 1:
                symptoms_present.append(feature_name)
            else:
                symptoms_absent.append(feature_name)


        symptoms_present_str = "Síntomas/factores PRESENTES: " + ", ".join(symptoms_present) if symptoms_present else "No hay síntomas presentes."
        symptoms_absent_str = "Síntomas/factores AUSENTES: " + ", ".join(symptoms_absent) if symptoms_absent else "Todos los síntomas están presentes."
        
        influential_factors_str = f"{symptoms_present_str}\n{symptoms_absent_str}"
        
        # Formatear datos de monitoreo si existen
        monitoring_context_str = ""
        if monitoring_data:
            monitoring_summary = []
            for entry in monitoring_data[:5]: # Limitar a los 5 más recientes
                details = [f"- Fecha: {entry.entry_date}"]
                if entry.temperature: details.append(f"Temperatura: {entry.temperature}°C")
                if entry.oxygen_level: details.append(f"Oxígeno: {entry.oxygen_level}%")
                if entry.heart_rate: details.append(f"Ritmo cardíaco: {entry.heart_rate} lpm")
                if entry.symptoms: details.append(f"Síntomas: {', '.join(entry.symptoms)}")
                if entry.overall_feeling: details.append(f"Sensación general: {entry.overall_feeling}/5")
                monitoring_summary.append("\n".join(details))
            
            if monitoring_summary:
                monitoring_context_str = (
                    "\n\n**Historial de Monitoreo Reciente del Paciente (últimos 5 registros):**\n"
                    + "\n\n".join(monitoring_summary)
                )

        prompt = f"""
        **Persona:** Eres un asistente de IA experto en monitoreo de salud y prevención de enfermedades. Tu función es generar recomendaciones estructuradas y extensas para un paciente que está monitoreando su estado de salud en casa.

        **Contexto General:** Un paciente ha recibido un resultado predictivo de COVID-19 de un modelo de machine learning. El resultado fue: **{prediction_label}**. Los factores que más influyeron en esta decisión, basados en la evaluación inicial, fueron:
        {influential_factors_str}
        {monitoring_context_str}

        **Tarea:** Basado en TODA la información disponible (resultado, factores influyentes e historial de monitoreo si existe), genera un array de **3 a 4 objetos JSON** para una sección de "Monitoreo y Recomendaciones". Cada objeto debe ofrecer un análisis detallado y consejos prácticos. Si hay datos de monitoreo, intégralos en el análisis para identificar tendencias o correlaciones. La explicación debe ser completa y profunda.

        **Formato:** La respuesta DEBE ser un array de objetos JSON válido, sin texto adicional. Cada objeto JSON debe tener la siguiente estructura:
        {{
          "icon": "string",
          "color": "string",
          "title": "string",
          "description": "string",
          "content": ["string", "string", ...]
        }}

        **Reglas Estrictas:**
        - El campo "icon" DEBE ser uno de los siguientes: "Stethoscope" (síntomas/análisis), "TrendingUp" (monitoreo/tendencias), "Shield" (prevención/defensas), "Heart" (condiciones/signos vitales), "Activity" (actividad/descanso), "BookOpen" (educación).
        - El campo "color" DEBE ser uno de los siguientes: "blue" (informativo), "green" (bienestar), "amber" (advertencia leve), "red" (atención importante), "purple" (consejo general).
        - El campo "content" debe ser un array con al menos dos strings, ofreciendo consejos claros, accionables y detallados.
        - El tono debe ser de apoyo, claro y educativo. Si hay datos de monitoreo, úsalos explícitamente en la explicación (ej. "Notamos que tu temperatura ha subido en los últimos días..."). Si no hay, da consejos más generales.

        **Ejemplo de respuesta JSON válida (para un caso positivo CON datos de monitoreo):**
        [
          {{
            "icon": "Stethoscope",
            "color": "red",
            "title": "Análisis de Síntomas y Tendencias",
            "description": "Correlacionando tu evaluación inicial con tu seguimiento diario.",
            "content": [
              "El resultado positivo se alinea con los síntomas que has estado reportando, como la fiebre y la tos. La presencia de 'dificultad para respirar' en tu evaluación inicial es un factor de alto impacto.",
              "En tu registro del {monitoring_data[0].entry_date if monitoring_data else 'ayer'}, tu temperatura fue de {monitoring_data[0].temperature if monitoring_data else '38.2'}°C. Es crucial vigilar si esta tendencia febril continúa o aumenta, ya que es un indicador clave de la actividad de la infección."
            ]
          }},
          {{
            "icon": "TrendingUp",
            "color": "amber",
            "title": "Recomendaciones de Monitoreo Activo",
            "description": "Qué y cómo debes vigilar en los próximos días.",
            "content": [
              "Continúa registrando tu temperatura y nivel de oxígeno dos veces al día. Un nivel de oxígeno consistentemente por debajo del 94% es una señal para buscar atención médica inmediata.",
              "Añade a tu registro tu frecuencia cardíaca en reposo. Un aumento notable y sostenido puede indicar que tu cuerpo está combatiendo la infección más intensamente."
            ]
          }},
          {{
            "icon": "Shield",
            "color": "green",
            "title": "Plan de Prevención y Cuidado",
            "description": "Medidas para protegerte a ti y a los demás.",
            "content": [
              "El factor de 'contacto con paciente COVID' fue muy influyente. Es vital que mantengas un aislamiento estricto para cortar la cadena de transmisión.",
              "Hidrátate constantemente y prioriza el descanso. Tu cuerpo necesita todos sus recursos para recuperarse eficazmente. Evita cualquier tipo de esfuerzo físico."
            ]
          }},
          {{
            "icon": "BookOpen",
            "color": "blue",
            "title": "Educación sobre Señales de Alarma",
            "description": "Cuándo es momento de contactar a un profesional.",
            "content": [
              "Aparte de la dificultad para respirar o el bajo nivel de oxígeno, otras señales de alarma incluyen dolor o presión persistente en el pecho, confusión o desorientación, y labios o rostro con tonalidad azulada.",
              "Ten a mano los números de emergencia y el contacto de tu centro de salud. No dudes en llamar si experimentas alguno de estos síntomas graves."
            ]
          }}
        ]
        """
        
        
        model = genai.GenerativeModel('gemini-1.5-flash')  # type: ignore
        response = model.generate_content(prompt)

       
        cleaned_response_text = response.text.strip().replace("`json", "").replace("`", "")
        explanation_json = json.loads(cleaned_response_text)
        
        return explanation_json

    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="La respuesta de la IA no fue un JSON válido.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ocurrió un error al generar la explicación: {e}")

