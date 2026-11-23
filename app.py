import os
import time
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

st.set_page_config(
    page_title="CampusAI • Academic Document Intelligence",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Modern SaaS Website Styling
st.markdown(
    """
    <style>
    /* Global Clean Typography & Spacing */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1150px;
    }

    /* Hero Website Banner */
    .campus-hero {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 55%, #172554 100%);
        padding: 2rem 2.25rem;
        border-radius: 18px;
        border: 1px solid rgba(99, 102, 241, 0.35);
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.45);
        margin-bottom: 1.5rem;
    }
    .campus-hero-tag {
        display: inline-block;
        background: rgba(56, 189, 248, 0.14);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.35);
        padding: 0.22rem 0.75rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 0.75rem;
    }
    .campus-hero h1 {
        color: #f8fafc;
        margin: 0;
        font-size: 2.05rem;
        font-weight: 800;
        letter-spacing: -0.02em;
    }
    .campus-hero p {
        color: #cbd5e1;
        margin: 0.5rem 0 0 0;
        font-size: 1rem;
        line-height: 1.5;
    }

    /* Status & Feature Cards */
    .feature-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 1.25rem;
        height: 100%;
    }
    .feature-card h4 {
        color: #f1f5f9;
        margin: 0 0 0.4rem 0;
        font-size: 1rem;
    }
    .feature-card p {
        color: #94a3b8;
        margin: 0;
        font-size: 0.88rem;
        line-height: 1.45;
    }

    /* Active Document Status Bar */
    .doc-status-bar {
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.35);
        border-radius: 12px;
        padding: 0.75rem 1.1rem;
        margin-bottom: 1.25rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        color: #d1fae5;
        font-size: 0.9rem;
    }

    /* Page Citation Badge */
    .page-badge {
        display: inline-block;
        background: rgba(99, 102, 241, 0.18);
        color: #a5b4fc;
        border: 1px solid rgba(99, 102, 241, 0.4);
        padding: 0.15rem 0.6rem;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 0.4rem;
    }

    /* Primary Button Polish */
    .stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #4f46e5 0%, #2563eb 100%);
        border: none;
        border-radius: 10px;
        font-weight: 600;
        padding: 0.55rem 1.2rem;
        transition: all 0.2s ease;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 18px rgba(79, 70, 229, 0.4);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize Session State
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0
if "page_count" not in st.session_state:
    st.session_state.page_count = 0
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = ""
if "quick_prompt" not in st.session_state:
    st.session_state.quick_prompt = ""

# Hero Header Banner
st.markdown(
    """
    <div class="campus-hero">
        <div class="campus-hero-tag">⚡ Retrieval-Augmented Generation (RAG) Portal</div>
        <h1>🎓 CampusAI — Academic Document Assistant</h1>
        <p>Instant, hallucination-free answers from your college syllabus, exam circulars, lab manuals, and lecture notes — backed by page-level source verification.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar: Workspace & Knowledge Base Ingestion
with st.sidebar:
    st.markdown("### 🔑 API Configuration")
    default_key = os.getenv("GOOGLE_API_KEY", "")
    if not default_key and "GOOGLE_API_KEY" in st.secrets:
        default_key = st.secrets["GOOGLE_API_KEY"]

    api_key = st.text_input(
        "Google AI Studio Key",
        type="password",
        value=default_key,
        placeholder="AIzaSy...",
        help="Free tier API key from Google AI Studio.",
    )
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
        st.success("Gemini Engine Connected", icon="✅")
    else:
        st.warning("Enter API Key to activate", icon="⚠️")

    st.markdown("[Get Free Gemini API Key ↗](https://aistudio.google.com/apikey)")
    st.divider()

    st.markdown("### 📂 Knowledge Base Upload")
    uploaded_pdf = st.file_uploader(
        "Upload Syllabus / Circular PDF",
        type=["pdf"],
        help="Upload any text-based academic PDF document.",
    )

    if st.button("⚡ Process & Index Document", type="primary", use_container_width=True):
        if not os.environ.get("GOOGLE_API_KEY"):
            st.error("Please enter your Google API Key first.")
        elif not uploaded_pdf:
            st.warning("Please select a PDF file to upload.")
        else:
            progress_bar = st.progress(10, text="Reading PDF pages...")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_pdf.read())
                tmp_path = tmp.name

            loader = PyPDFLoader(tmp_path)
            pages = loader.load()
            os.remove(tmp_path)

            progress_bar.progress(45, text="Splitting into semantic chunks...")
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=600,
                chunk_overlap=100,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
            chunks = splitter.split_documents(pages)

            progress_bar.progress(75, text="Generating Gemini vector embeddings...")
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-001",
                google_api_key=os.environ["GOOGLE_API_KEY"],
            )

            vector_db = None
            last_err = None
            for attempt in range(3):
                try:
                    vector_db = Chroma.from_documents(
                        documents=chunks,
                        embedding=embeddings,
                    )
                    break
                except Exception as e:
                    last_err = e
                    time.sleep(2)

            if vector_db is None:
                progress_bar.empty()
                st.error(f"Embedding failed after 3 attempts: {last_err}")
                st.stop()

            progress_bar.progress(100, text="Knowledge base ready!")
            time.sleep(0.4)
            progress_bar.empty()

            st.session_state.vector_store = vector_db
            st.session_state.chunk_count = len(chunks)
            st.session_state.page_count = len(pages)
            st.session_state.pdf_name = uploaded_pdf.name
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": f"Welcome! I have indexed **{uploaded_pdf.name}** ({len(pages)} pages across **{len(chunks)} semantic chunks**). Ask me any question about the document!",
                    "retrieved_chunks": [],
                    "latency": None,
                }
            ]
            st.rerun()

    if st.session_state.vector_store:
        st.divider()
        st.markdown("### 📊 Vector Store Telemetry")
        c1, c2 = st.columns(2)
        c1.metric("Pages", st.session_state.page_count)
        c2.metric("Chunks", st.session_state.chunk_count)
        if st.button("🗑️ Reset Conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    st.divider()
    st.caption("Architecture: PyPDF • Recursive Splitter • Gemini Embedding-001 • ChromaDB • Gemini 2.5 Flash")

# Helper Function to Execute RAG Query
def run_rag_query(question: str):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching ChromaDB vector index & synthesizing grounded response..."):
            t0 = time.time()
            retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
            docs = retriever.invoke(question)

            context_text = "\n\n".join(
                [f"[Page {d.metadata.get('page', 0) + 1}]\n{d.page_content}" for d in docs]
            )

            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0.0,
                google_api_key=os.environ["GOOGLE_API_KEY"],
            )
            prompt = ChatPromptTemplate.from_template(
                """You are CampusAI, an accurate Academic Document Assistant.
Answer the student's question strictly using ONLY the Context below.
If the answer is not found in the context, reply: "I cannot find that information in the uploaded document."
Always cite the page number(s) where the information was found, and format your answer cleanly with bullet points where helpful.

Context:
{context}

Question: {question}

Answer:"""
            )
            chain = prompt | llm
            response = chain.invoke({"context": context_text, "question": question})
            latency = round(time.time() - t0, 2)

            chunk_list = [
                {"page": d.metadata.get("page", 0) + 1, "text": d.page_content}
                for d in docs
            ]

        st.markdown(response.content)

        # Render Page Citation Badges + Latency
        pages_cited = sorted(list({ch["page"] for ch in chunk_list}))
        badges_html = "".join([f'<span class="page-badge">📄 Page {p}</span>' for p in pages_cited])
        st.markdown(f"{badges_html} <span style='color:#94a3b8; font-size:0.8rem;'>⏱️ {latency}s</span>", unsafe_allow_html=True)

        with st.expander("🔍 Inspect Top-3 Retrieved Vector Chunks"):
            for i, ch in enumerate(chunk_list, 1):
                st.markdown(f"**Chunk {i} — Page {ch['page']}**")
                st.info(ch["text"])

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response.content,
            "retrieved_chunks": chunk_list,
            "latency": latency,
        }
    )

# Main Workspace View
if not st.session_state.vector_store:
    # Onboarding Landing Cards (Website Style when no PDF is loaded yet)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            <div class="feature-card">
                <h4>1️⃣ Upload Academic PDF</h4>
                <p>Drop your Anna University syllabus, semester timetable, or exam circular into the sidebar uploader.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="feature-card">
                <h4>2️⃣ Semantic Vector Indexing</h4>
                <p>Text is split into 600-character overlapping chunks and embedded into ChromaDB using Gemini vectors.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div class="feature-card">
                <h4>3️⃣ Citation-Backed Q&A</h4>
                <p>Ask natural questions and verify every answer against the exact retrieved PDF chunks and page numbers.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("<br>", unsafe_allow_html=True)
    st.info("👈 **Get Started:** Enter your free Google AI Studio key and upload a PDF in the left sidebar.")
else:
    # Active Document Status Bar
    st.markdown(
        f"""
        <div class="doc-status-bar">
            <span>🟢 <b>Active Knowledge Base:</b> {st.session_state.pdf_name}</span>
            <span>📑 {st.session_state.page_count} Pages &nbsp;•&nbsp; 🧩 {st.session_state.chunk_count} Indexed Chunks</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Quick Suggested Questions Bar
    st.markdown("**💡 Quick Suggested Questions:**")
    q1, q2, q3 = st.columns(3)
    triggered_q = None
    if q1.button("📋 Summarize key topics & units", use_container_width=True):
        triggered_q = "Summarize the main subjects, units, and key topics covered in this document."
    if q2.button("📅 Check rules, dates & eligibility", use_container_width=True):
        triggered_q = "List all important rules, eligibility criteria, fees, or deadlines mentioned in this document."
    if q3.button("🎯 List important algorithms / concepts", use_container_width=True):
        triggered_q = "What are the core algorithms, technologies, or concepts highlighted in this document?"

    st.divider()

    # Render Existing Conversation
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("retrieved_chunks"):
                pages_cited = sorted(list({ch["page"] for ch in msg["retrieved_chunks"]}))
                badges_html = "".join([f'<span class="page-badge">📄 Page {p}</span>' for p in pages_cited])
                lat_str = f"⏱️ {msg['latency']}s" if msg.get("latency") else ""
                st.markdown(f"{badges_html} <span style='color:#94a3b8; font-size:0.8rem;'>{lat_str}</span>", unsafe_allow_html=True)

                with st.expander("🔍 Inspect Top-3 Retrieved Vector Chunks"):
                    for i, ch in enumerate(msg["retrieved_chunks"], 1):
                        st.markdown(f"**Chunk {i} — Page {ch['page']}**")
                        st.info(ch["text"])

    if triggered_q:
        run_rag_query(triggered_q)
        st.rerun()

# Chat Input Box
if user_input := st.chat_input("Ask a question about your uploaded document..."):
    if not os.environ.get("GOOGLE_API_KEY"):
        st.error("⚠️ Please enter your Google AI Studio API Key in the sidebar.")
    elif not st.session_state.vector_store:
        st.warning("⚠️ Please upload and index a PDF document in the sidebar first.")
    else:
        run_rag_query(user_input)
        st.rerun()