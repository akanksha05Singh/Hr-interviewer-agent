# HR Interviewer Agent 🤖

An AI-powered mock interview agent designed to conduct realistic technical and behavioral interviews, score candidate responses, and provide actionable feedback.

## 🌟 Features

-   **Role-Specific Interviews**: Supports Backend Engineer, Frontend Engineer, Data Scientist, and HR Generalist roles.
-   **Hybrid AI Engine**:
    -   **Gemini 1.5 Flash**: Generates dynamic questions and evaluates answers with a strict rubric.
    -   **Offline Fallback**: Robust local question bank ensures the interview never gets stuck, even without API access.
-   **Real-Time Scoring**: Provides immediate feedback, scores (0-10), and pass/fail verdicts.
-   **Voice & Text Support**: Candidates can speak their answers (transcribed via Speech-to-Text) or type them.
-   **Vector Store**: Local JSON-based vector store for RAG (Retrieval Augmented Generation)
-   **Speech**: Browser-based TTS and STT

## 🏗️ System Architecture

```mermaid
graph TD
    %% Nodes
    User((👤 Candidate))
    
    subgraph "Frontend Layer"
        UI[💻 Streamlit Interface]
        Voice[🎤 Voice Input]
        Text[⌨️ Text Input]
    end

    subgraph "Orchestration Layer"
        App{⚙️ Application Logic}
        Router[🔀 Mode Router]
    end

    subgraph "Intelligence Layer (The Brain)"
        Gemini[🧠 Gemini 1.5 Flash]
        Serp[🌐 SerpAPI Web Search]
        FAISS[📚 FAISS Local KB]
    end

    subgraph "Evaluation Engine (The Judge)"
        Scorer[⚖️ Hybrid Scorer]
        Embed[📐 OpenAI Embeddings]
        Fallback[🛡️ Offline Heuristics]
    end

    %% Connections
    User ==> UI
    UI --> Voice & Text
    Voice & Text ==> App
    
    App --> Router
    Router ==>|Online Mode| Gemini
    Router -.->|Offline Mode| Fallback
    
    Gemini <-->|Context| FAISS
    Gemini <-->|Fact Check| Serp
    
    Gemini ==>|Generated Q| App
    
    App ==> Scorer
    Scorer --> Gemini
    Scorer --> Embed
    
    Scorer ==>|Final Verdict| UI

    %% Styling
    classDef dark fill:#1a1a1a,stroke:#fff,stroke-width:2px,color:#fff;
    classDef blue fill:#1565c0,stroke:#fff,stroke-width:2px,color:#fff;
    classDef white fill:#fff,stroke:#1a1a1a,stroke-width:2px,color:#1a1a1a;
    classDef accent fill:#00e676,stroke:#1a1a1a,stroke-width:2px,color:#1a1a1a;

    class User,UI,Voice,Text white;
    class App,Router,Scorer,Embed,Fallback dark;
    class Gemini,Serp,FAISS blue;
    
    linkStyle default stroke:#333,stroke-width:2px;
```

## 🚀 Getting Started

1.  **Clone the repository**:
    ```bash
    git clone <your-repo-url>
    cd hr-interviewer-agent
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up Environment Variables**:
    Create a `.env` file in the root directory:
    ```env
    GOOGLE_API_KEY=your_gemini_api_key
    ```

4.  **Run the Application**:
    ```bash
    streamlit run app.py
    ```

## 🛡️ Robustness

This agent is built to be crash-proof:
-   **Multi-Model Fallback**: Retries with different Gemini models if one fails.
-   **Offline Mode**: Automatically switches to local questions if the internet or API is down.
-   **Auto-Recovery**: Detects and fixes "stuck" interview flows instantly.

## 📝 License

MIT
