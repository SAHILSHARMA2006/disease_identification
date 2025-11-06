import os
import json
import uvicorn
import numpy as np
import tensorflow as tf
from PIL import Image
from io import BytesIO
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse 
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from dotenv import load_dotenv

# --- INITIALIZATION AND CONFIGURATION ---

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found. Please set it in your .env file.")

client = genai.Client(api_key=API_KEY)
app = FastAPI(title="AGRIAI VISION API", version="1.0.0")

# --- MODEL CONFIGURATION ---
MODEL_PATH = "agrimind_model.keras" # Using .keras file
IMAGE_SIZE = (224, 224)
CLASS_NAMES = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___healthy',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Tomato___Leaf_Mold',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]
model = None 

# --- CORS ---
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Lifespan Event: Load Model at Startup ---
@app.on_event("startup")
async def load_ai_models():
    """Load the TensorFlow model when the FastAPI server starts."""
    global model
    try:
        if os.path.exists(MODEL_PATH):
            model = tf.keras.models.load_model(MODEL_PATH)
            print("TensorFlow .keras model loaded successfully.")
        else:
             print(f"WARNING: Model file '{MODEL_PATH}' not found. Prediction endpoint will fail.")
             model = None
    except Exception as e:
        print(f"Error loading TensorFlow model: {e}")
        model = None 

# --- Pydantic Schema (REMOVED local_summary_hindi) ---
class AgriculturalAdvice(BaseModel):
    disease_name: str = Field(description="The formal, common name of the disease or pest.")
    symptom_summary: str = Field(description="A brief description of key symptoms the farmer is seeing.")
    cause: str = Field(description="The root cause (e.g., Fungal pathogen, Bacteria, or Pest name).")
    confidence_score: float = Field(description="The ML model's confidence level (0.0 to 1.0) in the diagnosis.")
    severity_score: int = Field(description="A critical score from 1 (Low) to 10 (Critical) representing the urgency.")
    action_plan: list[str] = Field(description="A detailed, step-by-step mitigation plan covering chemical, organic, and irrigation/cultural solutions.")


# --- Image Prediction Helper ---
def classify_image(image_bytes: bytes):
    if model is None:
        return "Model Unavailable", 0.0
    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        image = image.resize(IMAGE_SIZE)
        img_array = np.array(image) / 255.0  
        img_array = np.expand_dims(img_array, axis=0)  

        predictions = model(tf.constant(img_array, dtype=tf.float32))
        probabilities = tf.nn.softmax(predictions[0])
        
        predicted_index = np.argmax(probabilities)
        confidence = float(probabilities[predicted_index])
        diagnosis = CLASS_NAMES[predicted_index]
        
        return diagnosis, confidence
    except Exception as e:
        print(f"Prediction failed: {e}")
        return "Prediction Error", 0.0


# --- *** NEW: ENDPOINT TO SERVE THE FRONTEND *** ---
@app.get("/", response_class=FileResponse)
async def get_root():
    """Serves the main index.html frontend file."""
    return "index.html"


# --- API Endpoint (NOW ACCEPTS LANGUAGE) ---
@app.post("/diagnose_image", response_model=AgriculturalAdvice)
async def diagnose_plant_image(
    file: UploadFile = File(...),
    symptoms_text: str = Form(""),
    language: str = Form("en") # Language field added
):
    
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Only JPEG and PNG are supported.")

    image_bytes = await file.read()

    # 1. Run Image Classification
    diagnosis, confidence = classify_image(image_bytes)
    
    if confidence == 0.0 and diagnosis == "Prediction Error":
         raise HTTPException(status_code=500, detail="Error during image classification. Check server logs.")

    if confidence == 0.0 and diagnosis == "Model Unavailable":
         raise HTTPException(status_code=503, detail="The ML model could not be loaded. Please ensure the 'agrimind_model.keras' file is in the root directory.")

    # 2. Prepare the LLM Prompt (NOW DYNAMIC)
    
    lang_instruction = "Respond ENTIRELY in English."
    if language == "hi":
        lang_instruction = "Respond ENTIRELY in Hindi (using Devanagari script)."

    text_grounding = ""
    if symptoms_text:
        text_grounding = f"The user provided this additional symptom description: '{symptoms_text}'. "
    
    user_query = (
        f"The computer vision model has identified the following plant issue: '{diagnosis}' "
        f"with a confidence score of {confidence:.2f}. "
        f"{text_grounding}"
        f"Based on ALL available information, provide the full disease name, symptom summary, cause, and a detailed, concise 3-step action plan. The action plan must include guidance on chemical pesticides, organic alternatives, and necessary irrigation/cultural changes. {lang_instruction}"
    )

    # 3. LLM Orchestration
    SYSTEM_PROMPT = (
        "You are AgriMind, a world-class, certified agricultural agronomist and pest control expert. "
        "Your task is to convert a raw ML diagnosis and user-provided symptoms into structured, actionable, and comprehensive advice. "
        "Your response MUST strictly adhere to the provided JSON schema."
    )

    try:
        payload = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=AgriculturalAdvice, 
        )

        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=[user_query],
            config=payload,
        )

        advice_data = json.loads(response.text)
        
        advice_data['confidence_score'] = confidence
        
        return AgriculturalAdvice(**advice_data)

    except Exception as e:
        print(f"Error during Gemini API call: {e}")
        raise HTTPException(status_code=500, detail="LLM Advisor failed to generate advice.")

# --- Run the application ---
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

