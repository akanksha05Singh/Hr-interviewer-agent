import os
import time
import json
from openai import OpenAI, RateLimitError, APIConnectionError, BadRequestError
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    client = OpenAI(api_key=api_key)
else:
    client = None
    print("Warning: OPENAI_API_KEY not found. OpenAI features will be disabled.")

def check_moderation(text: str) -> bool:
    """
    Checks if the input text violates OpenAI's content policy.
    Returns True if safe, False if flagged.
    """
    try:
        response = client.moderations.create(input=text)
        return not response.results[0].flagged
    except Exception as e:
        print(f"Moderation check failed: {e}")
        return True

def call_openai_chat(system_prompt: str, user_prompt: str, temperature=0.0):
    """
    Returns: (content, error_message)
    - content: The string response from LLM, or None if failed.
    - error_message: The error string if failed, or None if success.
    """
    # Guardrail: Check moderation
    if not check_moderation(user_prompt):
        print("Guardrail Alert: Input flagged by moderation API.")
        return None, "Input violated safety policies."

    messages = [{"role":"system","content":system_prompt},{"role":"user","content":user_prompt}]
    
    # Smart Fallback Strategy: Try Strong Model -> Fast Model
    models_to_try = ["gpt-4o", "gpt-4o-mini"]
    last_error = None
    
    for model in models_to_try:
        max_retries = 2
        base_delay = 1
        
        for attempt in range(max_retries):
            try:
                resp = client.chat.completions.create(
                    model=model, 
                    messages=messages, 
                    temperature=temperature, 
                    max_tokens=1000
                )
                return resp.choices[0].message.content, None
            except (RateLimitError, APIConnectionError) as e:
                last_error = str(e)
                # Check for quota specifically
                if "insufficient_quota" in str(e):
                    print(f"Quota Exceeded with {model}.")
                    return None, "insufficient_quota"
                    
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    print(f"OpenAI API Error ({model}): {e}. Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    print(f"OpenAI API Failed with {model}: {e}")
            except BadRequestError as e:
                print(f"Bad Request with {model}: {e}. Falling back immediately.")
                last_error = str(e)
                break
            except Exception as e:
                print(f"Unexpected Error with {model}: {e}")
                last_error = str(e)
                break 
        
        # If we are here, the current model failed. Loop continues to next model.
            
    return None, last_error
