# Enterprise Retrieval-Augmented Generation (RAG) System

A production-quality Enterprise RAG System built from scratch using only free and open-source tools. Users can upload multiple PDF documents, manage them in a vector database index, and ask natural language questions. The system retrieves relevant chunks using semantic search and generates accurate answers grounded exclusively in the uploaded documents, returning inline source citations and match percentages.

---

## 🏗️ Architecture

```mermaid
graph TB
    subgraph Frontend
        ST["Streamlit UI"]
    </style>
    subgraph Backend
        FA["FastAPI Server"]
        subgraph Services
            PL["PDF Loader"]
            CH["Chunker"]
            EM["Embedding Service"]
            RT["Retriever"]
            RP["RAG Pipeline"]
        end
        subgraph Storage
            VS["ChromaDB Vector Store"]
            FS["File System (data/)"]
        end
    end
    subgraph External
        GEM["Google Gemini API"]
        HF["HuggingFace Models"]
    end

    ST -->|HTTP Requests| FA
    FA --> PL --> CH --> EM --> VS
    FA --> RT --> VS
    RT --> RP --> GEM
    EM -->|sentence-transformers| HF
```

---

## ⚡ Tech Stack

*   **Core Logic:** Python 3.11+
*   **LLM Orchestrator:** LangChain (LangChain Express Language for clean pipeline chaining)
*   **Vector Database:** ChromaDB (local persistence, zero-infra setup)
*   **Local Embeddings:** Sentence Transformers (`all-MiniLM-L6-v2`, 384 dimensions)
*   **Generative AI Model:** Google Gemini 2.0 Flash (generous free-tier API)
*   **API Framework:** FastAPI (async execution, type safety, OpenAPI auto-docs)
*   **Frontend Interface:** Streamlit (customized glassmorphism design, sidebar file uploader, interactive chat)
*   **PDF Extractor:** PyPDF (native python parser, zero external C-dependencies)
*   **Deployment:** Docker & Docker Compose (multi-stage minimal build size)

---

## 🚀 Getting Started

### 📋 Prerequisites
*   Python 3.11 or higher installed.
*   A Google Gemini API key (Free tier). You can obtain one from [Google AI Studio](https://aistudio.google.com/apikey).

### 🔧 Local Installation & Setup

1.  **Clone or Navigate to the Directory:**
    ```bash
    cd "Enterprise RAG System"
    ```

2.  **Create and Activate a Virtual Environment:**
    ```bash
    python -m venv venv
    
    # On Windows:
    .\venv\Scripts\activate
    
    # On Linux/macOS:
    source venv/bin/activate
    ```

3.  **Install Required Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment Variables:**
    *   Rename the `.env` template (or edit the existing `.env`) in the root directory.
    *   Insert your Google Gemini API Key:
    ```ini
    GOOGLE_API_KEY=AIzaSyYourGeminiApiKeyHere...
    ```

5.  **Run the FastAPI Backend Server:**
    ```bash
    uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
    ```
    *   Swagger documentation is automatically available at: [http://localhost:8000/docs](http://localhost:8000/docs)

6.  **Run the Streamlit Web UI Application:**
    *   Open a new terminal window, activate the virtual environment, and run:
    ```bash
    streamlit run frontend/streamlit_app.py
    ```
    *   Open your browser at: [http://localhost:8501](http://localhost:8501)

---

## 🐳 Running with Docker

Run both the frontend and backend microservices using docker-compose. Volumes are used to ensure uploaded PDFs and the ChromaDB vector indexes are persisted on the host machine.

1.  **Build and Start Containers:**
    ```bash
    docker-compose up --build -d
    ```

2.  **Verify Contain Health:**
    *   FastAPI backend will run at [http://localhost:8000](http://localhost:8000)
    *   Streamlit UI will run at [http://localhost:8501](http://localhost:8501)

3.  **Shutdown and Remove Volumes (optional):**
    ```bash
    docker-compose down -v
    ```

---

## 🧪 Running Automated Tests

A suite of unit and integration tests is located under the `tests/` directory to verify document parsing, chunking, database insertion, and API controllers.

Run tests locally with pytest:
```bash
pytest tests/ -v
```

---

## 🛠️ Windows Memory & VM Optimization (Troubleshooting)

Windows environments with strict RAM limits or paging restrictions may trigger Out-Of-Memory (OOM) or loading crashes (`os error 1455` or thread pool failures). The following optimizations have been pre-applied in the codebase configurations to ensure reliability:

1.  **PyTorch Safetensors Paging Issue:**
    *   Loading sentence-transformers with default `.safetensors` can fail on constrained paging files. We bypass this by forcing standard PyTorch binary weights mapping in `backend/services/embeddings.py`:
        ```python
        model_kwargs = {'device': 'cpu', 'model_kwargs': {'use_safetensors': False}}
        ```

2.  **OpenBLAS Thread Allocations:**
    *   Multi-threading in linear algebra modules can trigger allocation crashes. We force single-thread bindings in `backend/config.py`:
        ```python
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["OMP_NUM_THREADS"] = "1"
        ```

3.  **ChromaDB File Locking:**
    *   Windows filesystem locks files in use. Clean up operations in `verify_*.py` are designed to retry or bypass locked sqlite resource errors.
