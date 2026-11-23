<div align="center">

# 🎓 CampusAI — Academic Document Intelligence Portal

**A Zero-Hallucination Retrieval-Augmented Generation (RAG) Assistant Built for College Syllabi, Exam Circulars & Lecture Notes**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-RAG_Orchestration-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://www.langchain.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Database-8B5CF6?style=for-the-badge)](https://www.trychroma.com/)
[![Gemini 2.5 Flash](https://img.shields.io/badge/Google_Gemini-2.5_Flash-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://ai.google.dev/)
[![Streamlit](https://img.shields.io/badge/Streamlit-SaaS_Portal-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-10B981?style=for-the-badge)](https://opensource.org/licenses/MIT)

---

### 📂 [Explore Source Code](#-project-structure) • 🏗️ [5-Step RAG Pipeline](#-system-architecture--5-step-rag-pipeline) • 🛠️ [Deep-Dive Tech Stack](#-technology-stack-deep-dive)

</div>

---

## 📖 Overview

**CampusAI** is an end-to-end **Retrieval-Augmented Generation (RAG)** web application engineered to transform static academic PDFs—such as university syllabi, examination circulars, lab manuals, and lecture notes—into an interactive, citation-backed conversational knowledge base.

Instead of relying on generic LLM pre-training data (which frequently hallucinates university-specific rules, attendance thresholds, or course units), CampusAI indexes uploaded documents into a **ChromaDB Vector Store** using **Google Gemini Embeddings (`models/gemini-embedding-001`)** and synthesizes strictly grounded answers via **Gemini 2.5 Flash (`temperature=0.0`)** with page-level source verification.

---

## 🎯 The Problem This Project Solves

Students and faculty deal with dozens of dense, unstructured PDF documents every semester—ranging from 80-page curriculum regulations to last-minute exam notices. Traditional search methods and ungrounded chatbots fail in four critical ways:

| Core Challenge | Why Traditional Tools Fail | How CampusAI Solves It |
| :--- | :--- | :--- |
| **📄 Manual PDF Ctrl+F Limitations** | Exact keyword search (`Ctrl+F`) fails when a student asks a conceptual question using different phrasing (e.g., searching *"exam fee rules"* when the circular says *"condonation charges"*). | **🧠 Dense Semantic Vector Search:** Converts text into high-dimensional embeddings so questions match by **meaning**, not just exact string overlap. |
| **⚠️ General LLM Hallucinations** | Asking standard ChatGPT or Gemini about college-specific regulations produces confident but fabricated rules, dates, or syllabus topics. | **🛡️ Strict Context-Only Guardrails:** Locks the LLM at `temperature=0.0` with explicit instructions to answer *only* from retrieved chunks or refuse if missing. |
| **❓ Lack of Verifiable Proof** | Even when an AI gives a correct summary, students cannot trust it during exam preparation without knowing where the rule came from. | **📌 Page-Level Provenance & Chunk Inspector:** Displays exact **`📄 Page X`** citation badges and lets users inspect the raw top-3 PDF chunks used to generate every reply. |
| **🔌 Transient Network Drops During Indexing** | Batch embedding API calls can occasionally drop HTTP/2 connections on unstable networks, crashing naive prototypes. | **🔄 Fault-Tolerant Retry Pipeline:** Implements an automatic 3-attempt retry loop with backoff during ChromaDB ingestion. |

---

## 🏗️ System Architecture & 5-Step RAG Pipeline

CampusAI separates document processing into an **Offline Ingestion Phase** and a **Real-Time Query & Synthesis Phase**:

    [1. Document Ingestion] ──▶ PyPDFLoader extracts raw text + page number metadata
               │
               ▼
    [2. Semantic Chunking]  ──▶ RecursiveCharacterTextSplitter (chunk_size=600, overlap=100)
               │
               ▼
    [3. Vector Embedding]   ──▶ Google gemini-embedding-001 + ChromaDB (with 3x Retry Guard)
               │
               ▼
    [4. Similarity Search]  ──▶ Cosine Similarity Retrieval fetches Top-K (k=3) relevant chunks
               │
               ▼
    [5. Grounded Synthesis] ──▶ Gemini 2.5 Flash (temp=0.0) outputs answer + Page Badges

---

## 🛠️ Technology Stack Deep-Dive

Every component in CampusAI was chosen to balance retrieval accuracy, low inference latency, and zero-cost deployment:

### 1. **LLM & Embedding Layer — Google Gemini API**
- **`models/gemini-embedding-001`:** Maps 600-character academic text chunks and student queries into dense vector representations capable of capturing domain-specific engineering and academic terminology.
- **`gemini-2.5-flash` (`temperature=0.0`):** Serves as the reasoning and synthesis engine. Setting `temperature=0.0` eliminates creative randomness, ensuring deterministic, factual adherence to the retrieved context.

### 2. **Vector Database — ChromaDB (`langchain-chroma`)**
- Acts as an in-memory/persistent vector index storing both the dense numerical embeddings and associated metadata (`page` number, document source).
- Executes fast **Cosine Similarity Search** (`k=3`) to isolate the three most contextually relevant document segments for every user query.

### 3. **RAG Orchestration & Document Processing — LangChain & PyPDF**
- **`PyPDFLoader` (`pypdf`):** Parses multi-page academic PDFs while preserving exact page indices so citations map accurately back to the source file.
- **`RecursiveCharacterTextSplitter`:** Splits documents hierarchically (`\n\n` → `\n` → `. `) with a `600`-character window and `100`-character overlap, preventing sentences or formulas from being severed across chunk boundaries.
- **`ChatPromptTemplate` & LCEL (`prompt | llm`):** Composes clean, modular prompt chains enforcing strict grounding and fallback refusal rules.

### 4. **Frontend Portal & Telemetry — Streamlit & Custom CSS**
- Built as a modern dark-themed SaaS portal featuring a gradient hero banner, onboarding workflow cards, live indexing progress bar, active knowledge-base status bar, one-click suggested question chips, and real-time response latency tracking.

---

## ✨ Key Features

- **⚡ Guided PDF Indexing Pipeline:** Visual step-by-step progress bar (`Reading PDF` → `Chunking` → `Generating Embeddings`) with automatic connection retry handling.
- **💡 One-Click Quick Prompts:** Built-in academic prompt chips to instantly summarize syllabus units, extract attendance/exam rules, or list core algorithms.
- **📌 Interactive Source Verification:** Every assistant response renders **`📄 Page X`** citation badges, response latency (`⏱️ Xs`), and an expandable **Top-3 Retrieved Vector Chunks** inspector.
- **💬 Stateful Chat Interface:** Maintains conversational flow with instant reset controls and live telemetry metrics (total pages and indexed vector chunks).

---

## 📂 Project Structure

```text
College-RAG-Chatbot/
├── 📄 app.py               # Complete SaaS-styled Streamlit UI & 5-step ChromaDB RAG pipeline
├── 📦 requirements.txt     # Pinned dependencies (LangChain, ChromaDB, Gemini SDK, Streamlit)
├── 🔐 .env.example         # Environment variable template for GOOGLE_API_KEY
├── 🙈 .gitignore           # Excludes virtual environments, secrets, and local vector cache
└── 📘 README.md            # Comprehensive architecture, problem statement & setup docs
```

---

## 💻 Local Setup & Installation

### 1️⃣ Clone the Repository

```bash
git clone [https://github.com/Rohithvasanthan/College-RAG-Chatbot.git](https://github.com/Rohithvasanthan/College-RAG-Chatbot.git)
cd College-RAG-Chatbot
```

### 2️⃣ Create & Activate Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate on Windows (PowerShell)
venv\Scripts\Activate.ps1

# Activate on Linux / macOS
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Configure Your Free Google Gemini API Key

1. Generate a free API key from **[Google AI Studio](https://aistudio.google.com/apikey)**.
2. Create your `.env` file:

```bash
cp .env.example .env
```

Add your key inside `.env`:

```env
GOOGLE_API_KEY=your_google_ai_studio_api_key_here
```

### 5️⃣ Launch the Application

```bash
streamlit run app.py
```

Open **`http://localhost:8501`** in your browser, upload any academic PDF, and start asking questions!

---

## 🚀 Future Roadmap (Scaling to Production & Hybrid Search)

- **Hybrid Retrieval (Dense Vector + BM25):** Combining ChromaDB semantic search with sparse keyword retrieval (`BM25`) via Reciprocal Rank Fusion to achieve 100% precision on exact subject codes (e.g., `CS3551`, `CCS375`).
- **Decoupled FastAPI Backend:** Exposing `/api/admin/upload` (JWT-protected for faculty) and `/api/student/ask` endpoints with persistent disk storage so students can query official college circulars directly from the college website.

---

## 👨‍💻 Author

**Rohithvasanthan E**  
*AI Engineer*  

[![GitHub](https://img.shields.io/badge/GitHub-Rohithvasanthan-181717?style=flat-square&logo=github)](https://github.com/Rohithvasanthan)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-rohithvasanthan18-0A66C2?style=flat-square&logo=linkedin)](https://linkedin.com/in/rohithvasanthan18)
[![YouTube](https://img.shields.io/badge/YouTube-Rohith_Flow-FF0000?style=flat-square&logo=youtube)](https://youtube.com/@rohithflow)#   C a m p u s A I  
 