import os
import logging
import json
import random
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logger = logging.getLogger("gemini_client")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(ch)
logger.setLevel(logging.INFO)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = "gemini-1.5-flash-001"

def get_gemini_client():
    if not GEMINI_API_KEY:
        return None
    return genai.Client(api_key=GEMINI_API_KEY)

def get_fallback_question(role):
    """Returns a fallback question if the API fails."""
    fallbacks = [
        {"question": f"Describe a challenging project you worked on as a {role} and how you overcame technical obstacles.", "reference": "Look for STAR method (Situation, Task, Action, Result) and specific technical details."},
        {"question": f"How do you handle conflicting priorities or tight deadlines in a {role} role?", "reference": "Candidate should mention communication, prioritization frameworks (e.g., Eisenhower matrix), and stakeholder management."},
        {"question": f"Explain a complex technical concept related to {role} to a non-technical stakeholder.", "reference": "Assess communication skills, clarity, and ability to avoid jargon."},
        {"question": "What are your strategies for ensuring code quality and maintainability?", "reference": "Expect mentions of code reviews, testing (unit/integration), CI/CD, and documentation."},
        {"question": "Tell me about a time you made a mistake in production. How did you handle it?", "reference": "Look for accountability, immediate remediation steps, and post-mortem/prevention strategies."}
    ]
    return random.choice(fallbacks)

def generate_question_gemini(role, difficulty="hard", history=[]):
    """
    Generates an interview question using Gemini 1.5 Flash.
    Returns a dict: {"question": "...", "reference": "..."}
    """
    models_to_try = ["gemini-1.5-flash-001", "gemini-1.5-flash", "gemini-1.0-pro", "gemini-pro"]
    
    for model_name in models_to_try:
        try:
            client = get_gemini_client()
            if not client:
                logger.error("Gemini API Key not found.")
                return get_fallback_question(role)

            history_str = "\n".join([f"- {h}" for h in history])
            
            # Use a simple schema for the output
            schema = {
                "type": "OBJECT",
                "properties": {
                    "question": {"type": "STRING", "description": "The technical or behavioral interview question."},
                    "reference": {"type": "STRING", "description": "A brief, 1-3 sentence expert answer or key concepts for the HR agent's reference."}
                },
                "required": ["question", "reference"]
            }
            
            prompt = f"""
            You are an expert HR interviewer. Generate one original, scenario-based, and challenging interview question for a candidate applying for the '{role}' role.
            
            - **Difficulty:** {difficulty}
            - **Length Constraint:** The entire question text MUST be under 120 words. Focus on the core challenge and be concise.
            - **Previous Questions (avoid repeating concepts):**
            {history_str}
            
            Generate the question and a brief reference answer (key concepts) for the evaluator.
            """
            
            logger.info(f"Attempting question generation with model: {model_name}")
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema,
                    temperature=0.7 
                )
            )
            
            if response.text:
                return json.loads(response.text)
        
        except Exception as e:
            logger.warning(f"Model {model_name} failed: {e}")
            continue # Try next model

    logger.error("All Gemini models failed. Using fallback.")
    return get_fallback_question(role)
