# HR Interviewer Agent 🤖

An AI-powered mock interview agent designed to conduct realistic technical and behavioral interviews, score candidate responses, and provide actionable feedback.

## 🌟 Features

-   **Role-Specific Interviews**: Supports Backend Engineer, Frontend Engineer, Data Scientist, and HR Generalist roles.
-   **Hybrid AI Engine**:
    -   **Gemini 1.5 Flash**: Generates dynamic questions and evaluates answers with a strict rubric.
    -   **Offline Fallback**: Robust local question bank ensures the interview never gets stuck, even without API access.
-   **Real-Time Scoring**: Provides immediate feedback, scores (0-10), and pass/fail verdicts.
-   **Voice & Text Support**: Candidates can speak their answers (transcribed via Speech-to-Text) or type them.
-   **Comprehensive Reporting**: Generates a detailed "True Report" with executive summary, strengths, and hiring recommendation.
-   **Robust Architecture**: Auto-recovery from API failures and "stuck" states.

## 🏗️ System Architecture

```mermaid
graph TD
    subgraph "User Interface (Streamlit)"
        Candidate[👤 Candidate]
        UI[💻 Web App]
        Voice[🎤 Voice Input]
        Text[⌨️ Text Input]
        AntiCheat[🛡️ Anti-Cheat Monitor]
    end

    subgraph "Orchestration Layer (Python)"
        App[⚙️ Main App Logic]
        Session[📝 Session State]
        Router[🔀 Mode Router]
    end

    subgraph "The Brain (Hybrid Engine)"
        direction TB
        subgraph "Cloud (Primary)"
            Gemini["🧠 Gemini 1.5 Flash"]
            Serp["🌐 SerpAPI (Web Search)"]
        end
        
        subgraph "Local (Fallback/RAG)"
            FAISS["📚 FAISS Vector Store"]
            LocalBrain["🤖 Local Heuristics"]
            OfflineQ["📂 Offline Question Bank"]
        end
    end

    %% Flows
    Candidate --> UI
    UI --> Voice & Text
    Voice & Text --> App
    AntiCheat -.->|Alerts| App
    
    App --> Router
    
    %% Online Flow
    Router -->|Online| Gemini
    Gemini -->|Generate Q| App
    Gemini -->|Evaluate A| App
    
    %% RAG Flow
    App -->|Retrieve Context| FAISS
    App -->|Fact Check| Serp
    FAISS -->|Context| Gemini
    Serp -->|Context| Gemini
    
    %% Offline Flow
    Router -->|Offline/Error| LocalBrain
    LocalBrain -->|Get Q| OfflineQ
    LocalBrain -->|Score A| App
    
    %% Output
    App -->|Feedback & Score| UI
    App -->|Final Report| UI
    
    classDef cloud fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef local fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;
    classDef ui fill:#fff3e0,stroke:#ef6c00,stroke-width:2px;
    
    class Gemini,Serp cloud;
    class FAISS,LocalBrain,OfflineQ local;
    class UI,Voice,Text,AntiCheat ui;
```

## 🛠️ Tech Stack

-   **Frontend**: Streamlit
-   **AI Models**: Google Gemini 1.5 Flash (via `google-genai` SDK)
-   **Vector Store**: Local JSON-based vector store for RAG (Retrieval Augmented Generation)
-   **Speech**: Browser-based TTS and STT

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
