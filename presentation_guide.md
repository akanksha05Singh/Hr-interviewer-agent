# Interview Presentation Guide: AI Interviewer Agent

This guide is designed to help you ace your presentation by explaining the technical details, design decisions, and a smooth demo flow for your "AI Interviewer Agent" project.

## 1. Architecture

You should describe the system as a **Hybrid AI Application** that combines Cloud LLMs with Local Processing for speed and reliability.

### High-Level Flow
1.  **Frontend (User Interface):** Built with **Streamlit** for a responsive, interactive web UI. It handles:
    *   Voice Recording (Microphone input)
    *   Text Input
    *   Real-time Feedback Display
    *   Anti-Cheat Monitoring (Tab switching detection)
2.  **Orchestration Layer (The Brain):**
    *   **Question Generation:** Uses **Gemini 1.5 Flash** to generate dynamic, role-specific questions based on chat history.
    *   **Context Retrieval (RAG):**
        *   **Local Knowledge Base:** Uses **FAISS** (Facebook AI Similarity Search) to retrieve expert definitions and concepts.
        *   **Web Search:** Uses **SerpAPI** (Google) to fetch real-time information if local data is missing.
        *   **Local Brain (Backup):** Includes a `sentence-transformers` module for fully offline embedding calculation if cloud APIs fail.
3.  **Evaluation Engine (The Judge):**
    *   **Primary Scorer:** **Gemini 1.5 Flash** evaluates answers against a strict JSON schema for Accuracy, Relevance, Depth, and Communication.
    *   **Secondary Scorer:** **OpenAI Embeddings** (`text-embedding-3-small`) calculate cosine similarity between the candidate's answer and expert reference material to detect semantic alignment.
    *   **Fallback Mechanism:** A heuristic-based "Offline Mode" ensures the demo never crashes, even if APIs are down.

### Architecture Diagram
```mermaid
graph TD
    User([User]) <--> UI[Streamlit UI<br/>(Voice/Text)]
    
    subgraph "Orchestration Engine"
        App[Python App Logic]
        LocalBrain[Local Brain<br/>(Backup)]
    end
    
    subgraph "AI Services (Cloud)"
        Gemini[Gemini 1.5 Flash<br/>(Gen/Eval/STT)]
        OAI[OpenAI Embeddings<br/>(Similarity)]
    end
    
    subgraph "RAG Knowledge Base"
        FAISS[(FAISS Vector DB)]
        Web[SerpAPI<br/>(Web Search)]
    end
    
    UI <--> App
    App --> Gemini
    App --> OAI
    App --> FAISS
    App --> Web
    App -.-> LocalBrain
    
    style User fill:#f9f,stroke:#333,stroke-width:2px
    style UI fill:#bbf,stroke:#333,stroke-width:2px
    style App fill:#dfd,stroke:#333,stroke-width:2px
    style Gemini fill:#fdd,stroke:#333,stroke-width:2px
```

### Tech Stack Summary
*   **Language:** Python 3.10+
*   **UI Framework:** Streamlit
*   **LLM (Logic & Scoring):** Google Gemini 1.5 Flash (Chosen for speed & large context window)
*   **Embeddings:** OpenAI `text-embedding-3-small` (with `sentence-transformers` backup)
*   **Vector DB:** FAISS (Local, fast, efficient)
*   **Voice:**
    *   **STT (Speech-to-Text):** **Gemini 1.5 Flash Multimodal** (Primary) - Processes audio directly without intermediate text conversion for higher accuracy. Falls back to Google Web Speech API.
    *   **TTS (Text-to-Speech):** Browser-native SpeechSynthesis API (Zero latency)

---

## 2. Reasoning & Design Choices

The Jury will ask *why* you built it this way. Here are the strong answers:

### Q: Why Streamlit instead of React/Next.js?
**A:** "For an AI-heavy application, **iteration speed is key**. Streamlit allows us to tightly couple the Python backend logic (RAG, Vector Search, LLM calls) with the frontend without managing complex API layers. It let us build a functional MVP in 48 hours."

### Q: Why Gemini 1.5 Flash?
**A:** "We needed a balance of **latency and cost**. Gemini 1.5 Flash is significantly faster and cheaper than GPT-4 while offering a massive context window, which allows us to pass the entire interview history into the prompt for better context awareness."

### Q: Why a Hybrid Evaluation (LLM + Embeddings)?
**A:** "LLMs can sometimes hallucinate or be charmed by eloquent but wrong answers. **Embeddings provide a mathematical 'ground truth' check.** If an answer sounds good (high LLM score) but has low semantic similarity to the expert reference (low Embedding score), the system flags it. This dual-check system reduces false positives."

### Q: How do you handle cheating?
**A:** "We implemented **browser-level event listeners** (JavaScript injection) to detect tab switching. If a candidate leaves the tab to Google an answer, the system logs a warning. We also enforce a strict time limit per question."

### Q: What if the Internet goes down during the demo?
**A:** "Reliability is critical. I built a robust **Offline Mode** that switches to a local question bank and heuristic scoring (based on keyword density and length) if the API calls fail. The show must go on."

---

## 3. Demo Walkthrough Script

Follow this flow for a "safe" and impressive demo.

**Step 1: Setup & Introduction (1 min)**
*   *Action:* Open the app. Show the Sidebar settings.
*   *Say:* "This is the AI Interviewer Agent. It's designed to conduct technical interviews for roles like Backend Engineer, Data Scientist, and more. It doesn't just ask questions; it listens, evaluates, and scores in real-time."

**Step 2: The "Happy Path" (2 mins)**
*   *Action:* Select **"Backend Engineer"**. Click **"Start Interview"**.
*   *Action:* The AI asks: "Tell me about yourself."
*   *Action:* **Use Voice Mode.** Click "Record Answer".
*   *Say (into mic):* "Hi, I'm a software engineer with 3 years of experience in Python and Cloud Architecture. I love building scalable systems."
*   *Action:* Submit.
*   *Observation:* Point out the **Real-time Transcription** and the **Instant Feedback** (Score: ~8/10, Verdict: PASS).

**Step 3: Technical Competence (2 mins)**
*   *Action:* The AI asks a technical question (e.g., "Explain ACID properties").
*   *Action:* **Intentionally give a mediocre answer.**
*   *Type:* "ACID stands for Atomicity and Durability. I forgot the rest."
*   *Action:* Submit.
*   *Observation:* Show how the AI gives a **lower score** and provides **Specific Improvements** (e.g., "You missed Consistency and Isolation"). This proves it's not just giving generic praise.

**Step 4: The "Anti-Cheat" & Final Report (1 min)**
*   *Action:* Briefly switch tabs and come back.
*   *Observation:* Show the "Warning: Tab switching detected" alert.
*   *Action:* Click **"End Session"**.
*   *Action:* Click **"Generate Comprehensive Report"**.
*   *Observation:* Show the final executive summary generated by Gemini.

---

## 4. Potential Jury Q&A

**Q: Can this replace human interviewers?**
**A:** "No, it's a **screening tool**. It filters out unqualified candidates efficiently so human interviewers can focus on the top 10% for deep-dive behavioral interviews."

**Q: How do you prevent prompt injection?**
**A:** "We use **strict output schemas** (JSON mode) to force the LLM to only return scores and feedback. We also strip code blocks from user input before sending it to the evaluation engine."

**Q: Is the scoring fair?**
**A:** "It's standardized. Unlike humans who might be tired or biased, the AI uses the exact same rubric and reference material for every candidate, ensuring consistency."

**Q: How scalable is this?**
**A:** "Since it uses stateless API calls and a local vector store, it can scale horizontally very easily. The vector store (FAISS) is highly optimized and can handle millions of documents with minimal latency."
