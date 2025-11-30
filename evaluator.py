import os, json, time, logging, math
from functools import lru_cache
from typing import Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

# ---- logging ----
logger = logging.getLogger("evaluator")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(ch)

# ---- external libs ----
try:
    from google import genai
    from google.genai import types
except Exception:
    genai = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# ---- env keys ----
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if OpenAI:
    # client will be initialized in get_embedding
    pass

# ---- utilities ----
def normalize_text(s: str) -> str:
    return " ".join(s.split()).strip()

def is_low_quality_answer(s: str) -> bool:
    if not s:
        return True
    t = s.strip().lower()
    if len(t.split()) < 4:
        return True
    for token in ["hahaha", "haha", "lol", "idk", "i don't know", "i dont know", "bye bye", "byebye"]:
        if token in t:
            return True
    # long repeated punctuation
    if len(t) > 200 and all(ch == t[0] for ch in t[:10]):
        return True
    return False

def cosine_similarity(a: list, b: list) -> float:
    dot = sum(x*y for x,y in zip(a,b))
    na = math.sqrt(sum(x*x for x in a)) or 1e-12
    nb = math.sqrt(sum(x*x for x in b)) or 1e-12
    cos = dot / (na*nb)
    cos = max(-1.0, min(1.0, cos))
    return (cos + 1.0) / 2.0

@lru_cache(maxsize=2048)
def cached_embedding(text: str, model: str = "text-embedding-3-large"):
    return get_embedding(text, model=model)

# ---- embedding provider ----
def get_embedding(text: str, model: str = "text-embedding-3-large") -> list:
    if not OpenAI or not OPENAI_API_KEY:
        raise RuntimeError("OpenAI client not available or OPENAI_API_KEY not set.")
    
    client = OpenAI(api_key=OPENAI_API_KEY)
    resp = client.embeddings.create(input=text, model=model)
    return resp.data[0].embedding

# ---- Gemini caller with Schema ----
def call_gemini_api_with_schema(prompt: str, model: str = "gemini-1.5-flash-001", temperature: float = 0.0) -> str:
    if not genai or not GOOGLE_API_KEY:
        raise RuntimeError("genai client not available or GOOGLE_API_KEY not set.")
    client = genai.Client(api_key=GOOGLE_API_KEY)
    
    # Define the strict schema for evaluation
    schema = {
        "type": "OBJECT",
        "properties": {
            "total_score": {"type": "NUMBER", "description": "Overall score from 0 to 10."},
            "verdict": {"type": "STRING", "enum": ["PASS", "FAIL"]},
            "summary": {"type": "STRING", "description": "1-2 sentence summary of the score and verdict."},
            "improvements": {
                "type": "ARRAY",
                "items": {"type": "STRING"},
                "description": "2-3 actionable points for improvement."
            }
        },
        "required": ["total_score", "verdict", "summary", "improvements"]
    }

    # google-genai SDK usage with response_schema
    resp = client.models.generate_content(
        model=model, 
        contents=prompt, 
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,
            temperature=temperature, 
            max_output_tokens=4000
        )
    )
    
    if hasattr(resp, "text") and resp.text:
        return resp.text
    return str(resp)

# ---- Strict evaluation prompt ----
EVAL_PROMPT = """
SYSTEM: You are a quick, expert technical interviewer. Your job is to score ONLY the candidate answer against the question and provided context. You MUST return a score from 0 to 10 and a concise verdict. Do NOT include greetings or extraneous text outside the JSON.

QUESTION:
{question}

CANDIDATE_ANSWER:
\"\"\"
{answer}
\"\"\"

RAG_CONTEXT (Expert Knowledge):
\"\"\"
{context}
\"\"\"

Provide the analysis and score. The final verdict is 'PASS' if score >= 6.0, otherwise 'FAIL'.
"""

# ---- Offline Fallback ----
def offline_evaluate(question: str, answer: str) -> Dict[str, Any]:
    """
    Provides a basic heuristic-based evaluation when the API is unavailable.
    """
    normalized = normalize_text(answer)
    words = len(normalized.split())
    
    # Basic scoring logic
    score = 0.0
    verdict = "FAIL"
    summary = "Offline Evaluation: "
    improvements = []
    
    if words < 10:
        score = 2.0
        summary += "Answer is too short."
        improvements.append("Expand on your answer with more details.")
    elif words < 30:
        score = 5.0
        summary += "Answer is brief but on topic."
        improvements.append("Provide specific examples to support your points.")
    else:
        score = 7.0
        verdict = "PASS"
        summary += "Good length. (Automated offline scoring)"
        improvements.append("Ensure you covered all technical constraints.")
        
    # Keyword bonus (very basic)
    keywords = ["design", "scale", "database", "api", "system", "consistency", "availability", "partition", "cache", "load", "balance"]
    hit_count = sum(1 for k in keywords if k in normalized.lower())
    score = min(10.0, score + (hit_count * 0.5))
    
    if score >= 6.0:
        verdict = "PASS"
        
    return {
        "status": "fallback",
        "total_score": round(score, 1),
        "verdict": verdict,
        "summary": summary,
        "improvements": improvements,
        "normalized_answer": normalized,
        "embedding_similarity": 0.0,
        "raw_preview": "OFFLINE_FALLBACK_MODE"
    }

# ---- Main evaluate function ----
def evaluate_with_gemini(question: str, candidate_answer: str, rag_context: str = "", embedding_kb_text: Optional[str] = None, thresholds: Optional[dict] = None) -> Dict[str, Any]:
    normalized = normalize_text(candidate_answer)
    logger.info("evaluate_called", extra={"q": question[:120], "preview": normalized[:200]})

    # prefilter
    if is_low_quality_answer(normalized):
        logger.info("prefilter_low_quality", extra={"preview": normalized[:200]})
        return {
            "status":"rejected",
            "total_score": 0.0,
            "verdict": "FAIL",
            "summary": "The answer was flagged as low quality (too short, nonsense, or irrelevant).",
            "improvements": ["Provide a more complete and professional answer.", "Avoid slang or one-word responses."],
            "embedding_similarity": 0.0,
            "normalized_answer": normalized,
            "raw_preview": "LOW_QUALITY_FILTER"
        }

    # embedding sim
    sim = 0.0
    # embedding sim
    sim = 0.0
    # if embedding_kb_text:
    #     try:
    #         # Check if OpenAI key is available before trying
    #         if not os.environ.get("OPENAI_API_KEY"):
    #             logger.warning("OPENAI_API_KEY not set. Skipping embedding similarity.")
    #             sim = 0.0
    #         else:
    #             emb_a = cached_embedding(normalized)
    #             emb_b = cached_embedding(embedding_kb_text)
    #             sim = cosine_similarity(emb_a, emb_b)
    #     except Exception as e:
    #         logger.warning(f"Embedding similarity check failed: {e}")
    #         sim = 0.0

    # call Gemini
    prompt = EVAL_PROMPT.format(question=question, answer=normalized, context=(rag_context or ""))
    raw = ""
    try:
        logger.info("calling_gemini_with_schema")
        raw = call_gemini_api_with_schema(prompt, model="gemini-1.5-flash-001", temperature=0.0)
        logger.info("gemini_raw_preview", extra={"raw": raw[:1600]})
    except Exception as e:
        logger.exception("gemini_call_failed")
        # FALLBACK TO OFFLINE EVALUATION
        logger.warning("Falling back to offline evaluation due to API error.")
        return offline_evaluate(question, normalized)

    # parse
    try:
        parsed = json.loads(raw)
        # Add embedding similarity to the result
        parsed["embedding_similarity"] = round(sim, 4)
        parsed["normalized_answer"] = normalized
        parsed["raw_preview"] = raw[:2000]
        
        return parsed
        
    except Exception as e:
        logger.exception("parse_failed")
        # FALLBACK TO OFFLINE EVALUATION ON PARSE ERROR TOO
        logger.warning("Falling back to offline evaluation due to parse error.")
        return offline_evaluate(question, normalized)

if __name__ == "__main__":
    # quick smoke test
    q = "Explain dependency injection."
    ans = "Dependency injection is a design pattern where dependencies are provided externally rather than created internally."
    kb = "Dependency injection is a pattern where dependencies are provided by external sources to improve testability."
    out = evaluate_with_gemini(q, ans, rag_context=kb, embedding_kb_text=kb)
    print(json.dumps(out, indent=2))
