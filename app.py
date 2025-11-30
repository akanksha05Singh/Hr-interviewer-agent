import os
from dotenv import load_dotenv
load_dotenv() # Load env vars before importing other modules that might use them

import streamlit as st
from vector_store import SimpleVectorStore
from rag_utils import build_rag_prompt, search_web
from llm_client import call_openai_chat, client # Kept for legacy/fallback or embeddings if needed
# from gemini_ollama_client import generate_question_ollama # Removed
from stt_tts import stt_from_file
from evaluator import evaluate_with_gemini, normalize_text # Import NEW evaluator
from local_brain import calculate_local_similarity # Local Backup Brain
import uuid, json, re
import tempfile
import streamlit.components.v1 as components
import datetime
import time
import random
from streamlit_autorefresh import st_autorefresh
import logging

# Setup logger
logger = logging.getLogger("app")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(ch)

st.set_page_config(page_title="HR Interviewer Agent", layout="wide")

# Check keys
# Check keys
if not os.getenv("GOOGLE_API_KEY") and not os.getenv("GEMINI_API_KEY"):
    st.error("CRITICAL: GOOGLE_API_KEY (or GEMINI_API_KEY) not found in .env file. Evaluation will fail.")
    st.stop()
if not os.getenv("OPENAI_API_KEY"):
    # st.warning("WARNING: OPENAI_API_KEY not found. Embedding similarity will be disabled.")
    logger.warning("OPENAI_API_KEY not found. Embedding similarity will be disabled.")

MAX_QUESTIONS = 10
MAX_ATTEMPTS = 2
QUESTION_DURATION = 120 # 2 minutes

def clean_json_output(text: str):
    # Remove markdown code blocks if present
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```", "", text)
    return text.strip()

def speak_text(text):
    # JavaScript for browser-based TTS
    safe_text = text.replace('"', '\\"').replace('\n', ' ')
    js = f"""
    <script>
        var msg = new SpeechSynthesisUtterance("{safe_text}");
        window.speechSynthesis.speak(msg);
    </script>
    """
    components.html(js, height=0, width=0)

# Anti-Cheat JS Injection
def inject_anti_cheat():
    js = """
    <script>
    document.addEventListener("visibilitychange", function() {
        if (document.hidden) {
            alert("WARNING: Tab switching is detected! This incident has been recorded.");
        }
    });
    </script>
    """
    components.html(js, height=0, width=0)

# Custom CSS for Pulse Animation & Timer
def inject_custom_css():
    st.markdown("""
    <style>
    .mic-pulse {
        display: inline-block;
        width: 20px;
        height: 20px;
        border-radius: 50%;
        background: #ff4b4b;
        box-shadow: 0 0 0 0 rgba(255, 75, 75, 1);
        transform: scale(1);
        animation: pulse 2s infinite;
        margin-right: 10px;
    }
    @keyframes pulse {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(255, 75, 75, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 10px rgba(255, 75, 75, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(255, 75, 75, 0); }
    }
    .timer-box {
        font-size: 1.5rem;
        font-weight: bold;
        color: #ff4b4b;
        border: 2px solid #ff4b4b;
        padding: 10px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

def inject_timer(remaining_seconds):
    # JS timer that starts from remaining_seconds
    if remaining_seconds < 0:
        remaining_seconds = 0
        
    js = f"""
    <script>
    function startTimer(duration, display) {{
        var timer = duration, minutes, seconds;
        var interval = setInterval(function () {{
            minutes = parseInt(timer / 60, 10);
            seconds = parseInt(timer % 60, 10);

            minutes = minutes < 10 ? "0" + minutes : minutes;
            seconds = seconds < 10 ? "0" + seconds : seconds;

            if (display) {{
                display.textContent = minutes + ":" + seconds;
                if (timer < 10) {{
                     display.style.color = "red";
                }}
            }}

            if (--timer < 0) {{
                clearInterval(interval);
                if (display) {{
                    display.textContent = "00:00 - Time's Up!";
                }}
            }}
        }}, 1000);
    }}

    window.onload = function () {{
        // Target the element in the parent document (Streamlit main window)
        var timeElement = window.parent.document.getElementById('time-display');
        if (timeElement) {{
            startTimer({remaining_seconds}, timeElement);
        }}
    }};
    </script>
    """
    components.html(js, height=0, width=0)
    
    # Calculate initial display
    mins = int(remaining_seconds / 60)
    secs = int(remaining_seconds % 60)
    time_str = f"{mins:02d}:{secs:02d}"
    st.markdown(f'<div class="timer-box">⏱️ <span id="time-display">{time_str}</span></div>', unsafe_allow_html=True)
    
    # Polling Strategy for Reliability
    current_qid = st.session_state.interview["qa"][-1]["id"] if st.session_state.interview["qa"] else "init"
    
    if remaining_seconds <= 10:
        st_autorefresh(interval=1000, limit=None, key=f"poll_fast_{current_qid}")
    else:
        st_autorefresh(interval=5000, limit=None, key=f"poll_slow_{current_qid}")

def get_offline_question(role, history):
    # Returns a dict with "question" and "reference"
    offline_pool = {
        "backend_engineer": [
            {"q": "Can you explain the difference between a process and a thread?", "ref": "A process is an instance of a program in execution with its own memory space. A thread is a unit of execution within a process, sharing memory with other threads."},
            {"q": "How do you handle database migrations in a production environment?", "ref": "Use migration tools (like Flyway or Alembic), backup data, test migrations in staging, use backward-compatible changes, and perform rolling updates."},
            {"q": "What strategies do you use to optimize a slow API endpoint?", "ref": "Profile code, optimize database queries (indexing), use caching (Redis), implement pagination, and use asynchronous processing."},
            {"q": "Explain the concept of dependency injection.", "ref": "Dependency Injection is a design pattern where dependencies are provided to a class rather than created within it, improving testability and decoupling."},
            {"q": "What are the pros and cons of microservices vs monolithic architecture?", "ref": "Microservices offer scalability and independent deployment but add complexity. Monoliths are simpler to develop initially but harder to scale and maintain as they grow."}
        ],
        "frontend_engineer": [
            {"q": "What is the difference between local storage, session storage, and cookies?", "ref": "Cookies are sent with requests and have expiration. LocalStorage persists until cleared. SessionStorage persists only for the session tab."},
            {"q": "Explain the concept of the Virtual DOM.", "ref": "Virtual DOM is a lightweight copy of the real DOM. React updates the Virtual DOM first, compares it with the previous version (diffing), and efficiently updates the real DOM."},
            {"q": "How do you ensure your application is accessible (a11y)?", "ref": "Use semantic HTML, ARIA labels, ensure keyboard navigability, use sufficient color contrast, and test with screen readers."},
            {"q": "What are some ways to optimize the performance of a React application?", "ref": "Use React.memo, useMemo, useCallback, code splitting (lazy loading), optimize images, and avoid unnecessary re-renders."},
            {"q": "Explain CSS Box Model.", "ref": "The Box Model consists of margins, borders, padding, and the actual content area. It determines the layout and size of elements."}
        ],
        "data_scientist": [
            {"q": "What is the difference between supervised and unsupervised learning?", "ref": "Supervised learning uses labeled data (e.g., classification). Unsupervised learning uses unlabeled data to find patterns (e.g., clustering)."},
            {"q": "How do you handle missing data in a dataset?", "ref": "Imputation (mean/median/mode), dropping rows/columns, using algorithms that handle missing values, or predicting missing values."},
            {"q": "Explain the bias-variance tradeoff.", "ref": "Bias is error from erroneous assumptions (underfitting). Variance is error from sensitivity to fluctuations (overfitting). You must balance them."},
            {"q": "What metrics do you use to evaluate a classification model?", "ref": "Accuracy, Precision, Recall, F1-Score, ROC-AUC curve."},
            {"q": "Describe a time you had to explain a complex model to a non-technical stakeholder.", "ref": "Focus on business impact, use analogies, avoid jargon, and visualize results."}
        ],
        "hr_generalist": [
            {"q": "How do you handle a conflict between two employees?", "ref": "Listen to both sides separately, identify the root cause, facilitate a mediation meeting, and establish a resolution plan."},
            {"q": "What is your approach to conducting an exit interview?", "ref": "Create a safe space, ask open-ended questions about reasons for leaving, gather feedback on culture/management, and use data to improve retention."},
            {"q": "How do you stay updated with labor laws and regulations?", "ref": "Subscribe to legal newsletters, attend HR seminars, network with peers, and consult with legal counsel."},
            {"q": "Describe a time you had to deliver difficult news to an employee.", "ref": "Be direct but compassionate, provide clear reasons, allow for questions, and offer support/next steps."},
            {"q": "What strategies do you use to improve employee engagement?", "ref": "Regular feedback, recognition programs, professional development opportunities, and fostering a positive work culture."}
        ]
    }
    
    pool = offline_pool.get(role, [{"q": "Tell me about your background.", "ref": "Candidate should discuss their education, relevant experience, skills, and career goals."}])
    
    available = [item for item in pool if item["q"] not in history]
    
    if not available:
        return {"q": f"Tell me more about your experience as a {role}.", "ref": "Candidate should elaborate on specific projects and achievements."}
        
    return random.choice(available)

def check_time_limit():
    if st.session_state.interview["qa"] and not st.session_state.finished:
        last_q = st.session_state.interview["qa"][-1]
        if not last_q.get("finalized") and st.session_state.q_start_time:
            elapsed = time.time() - st.session_state.q_start_time
            if elapsed > QUESTION_DURATION:
                st.toast("⏳ Time Limit Exceeded! Skipping...", icon="⚠️")
                
                last_q["finalized"] = True
                last_q["final_answer"] = "TIMEOUT (Time Limit Exceeded)"
                
                last_q["eval"] = {
                    "status": "rejected",
                    "ai_eval": {"total_score": 0, "breakdown": {"accuracy": 0, "relevance": 0, "depth": 0, "communication": 0}},
                    "embedding_similarity": 0.0,
                    "analysis": "Candidate failed to answer within the time limit.",
                    "improvements": "Please manage your time better."
                }
                
                if len(st.session_state.interview["qa"]) >= MAX_QUESTIONS:
                    st.session_state.finished = True
                else:
                    generate_question()
                
                st.rerun()

from gemini_client import generate_question_gemini # Add this

def generate_question():
    role = st.session_state.interview["role"]
    difficulty = "hard" 
    
    history = [q["question"] for q in st.session_state.interview["qa"]]
    
    st.write("DEBUG: generate_question called") 
    
    qtext = None
    ref_text = None

    # 1. Mandatory First Question: Introduction
    if not history:
        qtext = "Could you please introduce yourself and tell me a bit about your background?"
        ref_text = "Candidate should summarize their professional background, key skills, and motivation for this role. Look for clarity, confidence, and relevance."
    
    # 2. Subsequent Questions
    elif st.session_state.get("offline_mode"):
        st.info("Generating question in OFFLINE mode...")
        q_data = get_offline_question(role, history)
        qtext = q_data["q"]
        ref_text = q_data["ref"]
    else:
        try:
            with st.spinner(f"🧠 Generating next question..."):
                # Use Gemini for Question Generation
                try:
                    q_data = generate_question_gemini(role, difficulty, history) 
                except Exception as e:
                    st.error(f"Error generating question: {e}")
                    q_data = {}
                
                qtext = q_data.get("question")
                ref_text = q_data.get("reference")
                
                if not qtext:
                    st.warning("⚠️ Gemini Question Generation failed. Using Offline Mode fallback.")
                    q_data = get_offline_question(role, history) 
                    qtext = q_data["q"]
                    ref_text = q_data["ref"]
        except BaseException as e:
            logger.error(f"Error generating question: {e}")
            st.error(f"Error generating question: {e}")
            q_data = get_offline_question(role, history)
            qtext = q_data["q"]
            ref_text = q_data["ref"]

    # Append Logic
    if qtext:
        qid = str(uuid.uuid4())
        st.session_state.q_start_time = time.time()
        
        new_q = {
            "id":qid,
            "question":qtext,
            "reference": ref_text,
            "attempts": [], 
            "final_answer": None, 
            "finalized": False,
            "eval":None
        }
        
        # Explicitly reassign to ensure Streamlit detects change
        current_qa = st.session_state.interview["qa"]
        current_qa.append(new_q)
        st.session_state.interview["qa"] = current_qa
        
        logger.info(f"Generated question: {qtext}")
        st.write("DEBUG: Question appended successfully") 
    else:
        st.error("CRITICAL: Failed to generate question even with fallback.")


def retrieve_rag_context_for_role_and_question(role, question_text):
    context = ""
    # 1. Local Vector Store
    if "vstore" in st.session_state:
        try:
            retrieved_docs = st.session_state.vstore.query(question_text, k=1)
            if retrieved_docs:
                context += f"Local KB: {retrieved_docs[0]['text']}\n"
        except Exception:
            pass
    
    # 2. Web Search (if key exists)
    try:
        from rag_utils import search_web
        # User Request 5: Pass top snippets into rag_context
        web_res = search_web(question_text, num=3)
        if web_res:
            context += f"\n\nWebSearch:\n{web_res}\n"
    except Exception as e:
        logger.error(f"Web search error: {e}")
        
    return context if context else "No context available."

def on_submit_answer(question_text, user_answer, role):
    logger.info("answer_submitted -> evaluating", extra={"q": question_text[:120], "preview": user_answer[:200]})
    # load RAG context
    context = retrieve_rag_context_for_role_and_question(role, question_text)
    # call evaluator with full context
    eval_result = evaluate_with_gemini(question_text, user_answer, rag_context=context, embedding_kb_text=context)
    logger.info("parsed_eval", extra={"result": eval_result})
    return eval_result

# --- Load or create vector store (RAG)
VSTORE_PATH = "data/rag_store"
if "vstore" not in st.session_state:
    v = SimpleVectorStore()
    try:
        v.load(VSTORE_PATH)
        st.session_state.vstore = v
    except Exception:
        st.session_state.vstore = v

# session state for interview
if "interview" not in st.session_state:
    st.session_state.interview = {"role":"backend_engineer", "qa":[]}
if "finished" not in st.session_state:
    st.session_state.finished = False
if "q_start_time" not in st.session_state:
    st.session_state.q_start_time = None
if "offline_mode" not in st.session_state:
    st.session_state.offline_mode = False
if "trigger_finalize" not in st.session_state:
    st.session_state.trigger_finalize = False

# DEBUG STATE
# st.write(f"DEBUG STATE: trigger_finalize={st.session_state.trigger_finalize}, finished={st.session_state.finished}")

# Inject Anti-Cheat & CSS
inject_anti_cheat()
inject_custom_css()

# UI: Sidebar
with st.sidebar:
    st.title("HR Agent")
    st.caption("Powered by Gemini 1.5 & Llama 3")
    
    st.markdown("### 1. Settings")
    disabled = len(st.session_state.interview["qa"]) > 0
    
    st.session_state.interview["role"] = st.selectbox("Role", ["backend_engineer","frontend_engineer","data_scientist","hr_generalist"], index=0, disabled=disabled)
    
    if st.button("Load Sample Knowledge Base", disabled=disabled):
        # Expanded sample data
        sample_docs = [
            {"id": "db_acid", "text": "ACID properties in databases stand for Atomicity, Consistency, Isolation, Durability.", "meta": {"topic": "database"}},
            {"id": "db_index", "text": "Database indexing improves the speed of data retrieval operations on a database table.", "meta": {"topic": "database"}},
            {"id": "db_norm", "text": "Normalization is the process of organizing data in a database to reduce redundancy.", "meta": {"topic": "database"}},
            {"id": "api_rest", "text": "REST API design principles include statelessness, cacheability, and layered system.", "meta": {"topic": "api"}},
            {"id": "api_graphql", "text": "GraphQL is a query language for APIs that allows clients to request exactly the data they need.", "meta": {"topic": "api"}},
            {"id": "api_auth", "text": "OAuth 2.0 is an authorization framework that enables applications to obtain limited access to user accounts.", "meta": {"topic": "security"}},
            {"id": "ml_overfit", "text": "Overfitting in machine learning happens when a model learns the training data too well, including noise.", "meta": {"topic": "ml"}},
            {"id": "ml_bias", "text": "Bias-variance tradeoff is the property of a model that the variance of the parameter estimated across samples can be reduced by increasing the bias in the estimated parameters.", "meta": {"topic": "ml"}},
            {"id": "fe_react", "text": "React hooks allow you to use state and other React features without writing a class.", "meta": {"topic": "frontend"}},
            {"id": "fe_dom", "text": "The Virtual DOM is a programming concept where an ideal, or 'virtual', representation of a UI is kept in memory and synced with the 'real' DOM.", "meta": {"topic": "frontend"}},
            {"id": "sys_cap", "text": "CAP theorem states that a distributed data store can only provide two of the following three guarantees: Consistency, Availability, Partition Tolerance.", "meta": {"topic": "system_design"}},
            {"id": "sys_lb", "text": "Load balancing refers to the process of distributing a set of tasks over a set of resources, with the aim of making their overall processing more efficient.", "meta": {"topic": "system_design"}},
            {"id": "hr_star", "text": "The STAR method (Situation, Task, Action, Result) is a structured manner of responding to behavioral interview questions.", "meta": {"topic": "hr"}},
            {"id": "hr_conflict", "text": "Conflict resolution in the workplace involves communication, active listening, and finding a compromise.", "meta": {"topic": "hr"}}
        ]
        st.session_state.vstore.add_documents(sample_docs)
        st.session_state.vstore.save(VSTORE_PATH)
        st.success(f"Loaded {len(sample_docs)} sample documents!")

    st.markdown("---")
    st.markdown("### 2. Session Controls")
    
    # Force Offline Mode Toggle
    st.session_state.offline_mode = st.checkbox("Force Offline Mode (Use if API is stuck)", value=st.session_state.get("offline_mode", False))
    
    col_reset, col_finish = st.columns(2)
    with col_reset:
        restart_disabled = len(st.session_state.interview["qa"]) > 0 and not st.session_state.finished
        if st.button("Start New Interview", type="primary", disabled=restart_disabled):
            st.session_state.interview["qa"] = []
            st.session_state.finished = False
            st.session_state.q_start_time = None
            # st.session_state.offline_mode = False # Don't reset this if user wants it on
            st.rerun()
            
    with col_finish:
        if len(st.session_state.interview["qa"]) > 0 and not st.session_state.finished:
            if st.button("End Session"):
                st.session_state.finished = True
                st.rerun()

    st.markdown("---")
    with st.expander("ℹ️ How does scoring work?"):
        st.markdown("""
        **Hybrid AI Engine:**
        
        1. **Gemini 1.5 Flash**: Evaluates your answer against a strict rubric (Accuracy, Relevance, Depth, Communication).
        2. **Embedding Similarity**: Checks mathematical alignment with expert knowledge.
        3. **Llama 3**: Generates challenging, role-specific questions locally.
        """)
        
    if st.session_state.get("offline_mode"):
        st.warning("⚠️ **Offline Mode Active**: Using pre-defined questions and local scoring.")

st.title("HR Interviewer Agent")

# Enforce Time Limit on every run
check_time_limit()

# Progress Bar
progress = len(st.session_state.interview["qa"]) / MAX_QUESTIONS
st.progress(progress, text=f"Interview Progress: {int(progress*100)}%")


col1, col2 = st.columns([2,1])

with col1:
    st.write("### Interview Session")
    st.caption("ℹ️ Questions are generated by Gemini 1.5 Flash.")
    
    # Chat History
    for i, q in enumerate(st.session_state.interview["qa"]):
        with st.chat_message("assistant"):
            st.write(f"**Q{i+1}:** {q['question']}")
            if i == len(st.session_state.interview["qa"]) - 1 and not q.get("finalized") and not st.session_state.finished:
                 speak_text(q['question'])
        
        if q.get("finalized"):
            with st.chat_message("user"):
                st.write(q["final_answer"])
                if len(q["attempts"]) > 1:
                    st.caption(f"(Finalized after {len(q['attempts'])} attempts)")
            
            if q.get("eval"):
                st.write(f"DEBUG: eval keys: {list(q['eval'].keys())}") # DEBUG
                st.write(f"DEBUG: verdict: {q['eval'].get('verdict')}") # DEBUG
                with st.expander(f"Evaluation for Q{i+1}", expanded=True):
                    # Gemini Evaluation Data
                    # NEW STRUCTURE: eval_result has keys: status, ai_eval, embedding_similarity
                    eval_result = q["eval"]
                    status = eval_result.get("status", "unknown")
                    
                    # Extract values from flattened schema
                    score = eval_result.get("total_score", 0)
                    verdict = eval_result.get("verdict", "N/A")
                    summary = eval_result.get("summary", "No summary available.")
                    improvements = eval_result.get("improvements", [])
                    
                    sim_score = eval_result.get("embedding_similarity", 0.0)
                    
                    col_score, col_metrics = st.columns([1, 2])
                    
                    with col_score:
                        st.metric("Total Score", f"{score:.1f}/10")
                        # Display the final verdict prominently
                        if verdict == "PASS":
                            st.success(f"**Final Verdict: {verdict}**")
                        elif verdict == "FAIL":
                            st.error(f"**Final Verdict: {verdict}**")
                        else:
                            st.warning(f"**Final Verdict: {verdict}**")
                        
                        st.metric("Embedding Similarity", f"{sim_score:.2f}")
                        
                    with col_metrics:
                        st.markdown("**Analysis:**")
                        st.info(f"{summary}")

                    st.markdown("---")
                    
                    if improvements:
                        if isinstance(improvements, list):
                            st.success(f"**💡 Improvements:**\n" + "\n".join([f"- {i}" for i in improvements]))
                        else:
                            st.success(f"**💡 Improvements:**\n{improvements}")

    # Current Action Area
    if not st.session_state.interview["qa"] and not st.session_state.finished:
        if st.button("Start Interview", type="primary"):
            generate_question()
            st.rerun()
    elif not st.session_state.finished:
        last_q = st.session_state.interview["qa"][-1]
        
        if last_q.get("finalized") and len(st.session_state.interview["qa"]) < MAX_QUESTIONS:
             # AUTOMATIC RECOVERY: No excuses, just generate the next question.
             # We use offline fallback here to GUARANTEE it works instantly.
             role = st.session_state.interview["role"]
             history = [q["question"] for q in st.session_state.interview["qa"]]
             q_data = get_offline_question(role, history)
             
             qid = str(uuid.uuid4())
             st.session_state.q_start_time = time.time()
             st.session_state.interview["qa"].append({
                "id":qid,
                "question":q_data["q"],
                "reference": q_data["ref"],
                "attempts": [], 
                "final_answer": None, 
                "finalized": False,
                "eval":None
             })
             st.rerun()
        
        elif not last_q.get("finalized"):
            st.write("---")
            
            if st.session_state.q_start_time:
                elapsed = time.time() - st.session_state.q_start_time
                remaining = int(QUESTION_DURATION - elapsed)
                inject_timer(remaining)
            else:
                inject_timer(120)
            
            attempts_count = len(last_q["attempts"])
            st.write(f"**Provide your answer (Attempt {attempts_count + 1} of {MAX_ATTEMPTS}):**")
            
            current_draft = last_q["attempts"][-1]["text"] if last_q["attempts"] else None
            
            if current_draft:
                st.info(f"**Current Draft:** {current_draft}")
                
                col_fin, col_retry = st.columns([1,1])
                with col_fin:
                    # Session state trigger pattern to handle refreshes
                    if st.button("Finalize Answer (Debug)", type="primary"):
                        st.session_state.trigger_finalize = True
                    
                    if st.session_state.get("trigger_finalize"):
                        st.write("DEBUG: Trigger detected, starting evaluation...")
                        try:
                            with st.spinner("Evaluating answer with Gemini 1.5..."):
                                # 1. Call Evaluation
                                eval_result = on_submit_answer(last_q["question"], current_draft, st.session_state.interview["role"])
                                
                                # 2. CRITICAL DEBUG BLOCK
                                st.markdown("### 🚨 DEBUG: Raw Evaluation Result")
                                try:
                                    st.code(json.dumps(eval_result, indent=2), language="json") 
                                except Exception as e:
                                    st.error(f"FATAL: Evaluation result object is not a valid Python dictionary: {e}")
                                    st.code(str(eval_result))

                                # 3. Handle Errors and Update State
                                if eval_result.get("status") in ["error", "flagged"]:
                                     st.error(f"Evaluation API Failed: {eval_result.get('error') or eval_result.get('parse_error')}")
                                     
                                st.session_state.interview["qa"][-1]["eval"] = eval_result
                                
                                # 4. Finalize Question and Advance Flow
                                last_q["finalized"] = True
                                last_q["final_answer"] = current_draft
                                
                                # Reset trigger
                                st.session_state.trigger_finalize = False
                                
                                if len(st.session_state.interview["qa"]) >= MAX_QUESTIONS:
                                    st.session_state.finished = True
                                else:
                                    generate_question()
                                    
                            # Wait for user to see the debug output before continuing
                            st.success("Evaluation complete. Please check the debug output above.")
                            time.sleep(2) # Short pause to see result
                            st.rerun()

                        except Exception as e:
                            st.error(f"FATAL EXECUTION ERROR: {e}")
                            st.session_state.trigger_finalize = False # Reset on error
            
            if attempts_count < MAX_ATTEMPTS:
                tab1, tab2 = st.tabs(["🎤 Voice", "⌨️ Text"])
                
                new_attempt_text = None
                skipped = False
                
                with tab1:
                    st.markdown('<div class="mic-pulse"></div> <span style="vertical-align:super; font-weight:bold;">Live Recording</span>', unsafe_allow_html=True)
                    audio_value = st.audio_input("Record Answer", key=f"mic_{len(st.session_state.interview['qa'])}_{attempts_count}")
                    
                    transcript_key = f"transcript_{len(st.session_state.interview['qa'])}_{attempts_count}"
                    if transcript_key not in st.session_state:
                         st.session_state[transcript_key] = ""

                    if audio_value:
                        if st.button("Submit Voice Answer", key=f"sub_voice_{transcript_key}"):
                            with st.spinner("Processing & Transcribing..."):
                                try:
                                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                                        audio_value.seek(0)
                                        tmp.write(audio_value.read())
                                        tmp_path = tmp.name
                                    
                                    # Transcribe
                                    transcribed = stt_from_file(tmp_path)
                                    os.unlink(tmp_path)
                                    
                                    if transcribed.strip():
                                        # ATOMIC SUBMISSION: Process immediately
                                        st.success(f"Submitted: {transcribed}")
                                        st.toast("✅ Voice Answer Submitted! Evaluating...", icon="🚀")
                                        logger.info("received voice draft (manual-submit)")
                                        time.sleep(1) # Brief pause to show success message
                                        
                                        # 1. Append to history (Draft)
                                        current_q = st.session_state.interview["qa"][-1]
                                        current_q["attempts"].append({
                                            "text": transcribed,
                                            "timestamp": datetime.datetime.now().strftime("%H:%M:%S")
                                        })
                                        
                                        # 2. Trigger Finalization (Reuses main logic)
                                        # This avoids double-evaluation and ensures consistent flow.
                                        st.session_state.trigger_finalize = True
                                        
                                        # 3. Force Rerun
                                        st.rerun()
                                    else:
                                        st.error("Could not hear anything. Please try again.")
                                except Exception as e:
                                    st.error(f"Error processing audio: {e}")

                with tab2:
                    text_input = st.text_area("Type your answer here", key=f"text_{len(st.session_state.interview['qa'])}_{attempts_count}")
                    col_sub, col_skip = st.columns([1,1])
                    with col_sub:
                        if st.button("Submit Text Draft"):
                            new_attempt_text = text_input
                            logger.info("received draft")
                    with col_skip:
                        if st.button("Skip Question"):
                            new_attempt_text = "SKIPPED"
                            skipped = True

                if new_attempt_text:
                    last_q["attempts"].append({
                        "text": new_attempt_text,
                        "timestamp": datetime.datetime.now().strftime("%H:%M:%S")
                    })
                    
                    if skipped:
                        last_q["finalized"] = True
                        last_q["final_answer"] = "SKIPPED"
                        eval_result = {
                            "status": "rejected",
                            "reason": "Skipped by user",
                            "ai_eval": {},
                            "embedding_similarity": 0.0
                        }
                        st.session_state.interview["qa"][-1]["eval"] = eval_result
                        
                        if len(st.session_state.interview["qa"]) >= MAX_QUESTIONS:
                            st.session_state.finished = True
                        else:
                            generate_question()
                    
                    st.rerun()

with col2:
    if st.session_state.finished:
        st.write("### 📊 Interview Summary")
        
        per_q = []
        total_score_sum = 0
        valid_scores_count = 0
        
        for q in st.session_state.interview["qa"]:
            if q.get("eval"):
                eval_data = q["eval"]
                # Handle flattened schema
                score = eval_data.get("total_score", 0)
                verdict = eval_data.get("verdict", "N/A")
                summary = eval_data.get("summary", "")
                
                per_q.append({
                    "question": q["question"],
                    "answer": q.get("final_answer",""),
                    "score": score,
                    "verdict": verdict,
                    "summary": summary
                })
                
                if isinstance(score, (int, float)):
                    total_score_sum += score
                    valid_scores_count += 1
        
        if per_q:
            # Calculate Average based on TOTAL QUESTIONS (MAX_QUESTIONS)
            # User request: "give total score on basis of 10 question even if the interview submitted earlier"
            final_avg = total_score_sum / MAX_QUESTIONS 
            
            col_res1, col_res2 = st.columns(2)
            with col_res1:
                st.metric("Overall Score", f"{final_avg:.1f}/10")
            with col_res2:
                pass_count = sum(1 for item in per_q if item["verdict"] == "PASS")
                st.metric("Pass Rate", f"{pass_count}/{len(per_q)} (Attempted)")
            
            st.markdown("### Detailed Breakdown")
            for i, item in enumerate(per_q):
                with st.expander(f"Q{i+1}: {item['verdict']} ({item['score']}/10)"):
                    st.write(f"**Question:** {item['question']}")
                    st.write(f"**Answer:** {item['answer']}")
                    st.info(f"**Feedback:** {item['summary']}")

            # Generate Comprehensive Report
            if st.button("Generate Comprehensive Report (AI)"):
                with st.spinner("Compiling final report..."):
                    try:
                        # Prepare context for the LLM
                        report_context = "\n\n".join([
                            f"Q{i+1}: {item['question']}\nCandidate Answer: {item['answer']}\nScore: {item['score']}\nVerdict: {item['verdict']}\nFeedback: {item['summary']}"
                            for i, item in enumerate(per_q)
                        ])
                        
                        from gemini_client import get_gemini_client
                        client = get_gemini_client()
                        if client:
                            prompt = f"""
                            You are a Senior Hiring Manager. Review the following interview transcript and provide a final hiring recommendation.
                            
                            Role: {st.session_state.interview['role']}
                            
                            TRANSCRIPT:
                            {report_context}
                            
                            OUTPUT FORMAT:
                            ## Executive Summary
                            [2-3 sentences]
                            
                            ## Key Strengths
                            - [Point 1]
                            - [Point 2]
                            
                            ## Areas for Improvement
                            - [Point 1]
                            - [Point 2]
                            
                            ## Final Recommendation
                            [HIRE / NO HIRE / STRONG HIRE] - [Justification]
                            """
                            
                            response = client.models.generate_content(
                                model="gemini-1.5-flash",
                                contents=prompt
                            )
                            st.markdown("---")
                            st.markdown(response.text)
                        else:
                            st.error("Cannot generate AI report: API Key missing.")
                    except Exception as e:
                        st.error(f"Failed to generate report: {e}")

            
            if st.button("Export Results JSON"):
                export_data = {
                    "interview": st.session_state.interview, 
                    "summary": {
                        "average_score": final_avg,
                        "questions_count": len(per_q)
                    }
                }
                st.download_button("Download JSON",
                                   data=json.dumps(export_data, indent=2),
                                   file_name="interview_result.json")
        else:
            st.warning("Interview ended early. No questions were completed.")
            if st.button("Start New Interview", key="restart_main"):
                st.session_state.interview["qa"] = []
                st.session_state.finished = False
                st.session_state.q_start_time = None
                # st.session_state.offline_mode = False
                st.rerun()
                
    elif st.session_state.interview["qa"]:
        st.info(f"Question {len(st.session_state.interview['qa'])} of {MAX_QUESTIONS}")
