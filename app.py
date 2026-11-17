import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="AI Healthcare Research Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------- CUSTOM CSS STYLING ----------------
st.markdown(
    """
    <style>
    /* Main container styling */
    .main .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* Hero Header Banner */
    .hero-banner {
        background: linear-gradient(135deg, #0e7490 0%, #0369a1 50%, #1d4ed8 100%);
        padding: 24px 30px;
        border-radius: 16px;
        color: #ffffff;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(14, 116, 144, 0.3);
    }
    .hero-title {
        font-size: 2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
        color: #ffffff !important;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .hero-subtitle {
        font-size: 1rem;
        margin-top: 6px;
        margin-bottom: 14px;
        color: #e0f2fe;
        opacity: 0.95;
    }
    .hero-badges {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 8px;
    }
    .hero-badge {
        background: rgba(255, 255, 255, 0.18);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.28);
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.78rem;
        font-weight: 600;
        color: #ffffff;
        letter-spacing: 0.2px;
    }

    /* Metric cards */
    .metric-card {
        background: rgba(14, 165, 233, 0.05);
        border: 1px solid rgba(14, 165, 233, 0.22);
        border-radius: 12px;
        padding: 12px 14px;
        text-align: center;
        margin-bottom: 8px;
    }
    .metric-num {
        font-size: 1.5rem;
        font-weight: 700;
        color: #0284c7;
        line-height: 1.1;
    }
    .metric-lbl {
        font-size: 0.75rem;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 2px;
    }

    /* Result & Response Card */
    .response-card {
        background: rgba(248, 250, 252, 0.9);
        border: 1px solid #e2e8f0;
        border-left: 5px solid #0284c7;
        border-radius: 10px;
        padding: 20px;
        margin: 16px 0;
    }

    /* Source Citation Card */
    .source-box {
        background: rgba(241, 245, 249, 0.8);
        border: 1px solid #cbd5e1;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 10px;
        font-size: 0.88rem;
    }
    .source-badge {
        background: #0284c7;
        color: white;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 6px;
    }

    /* Suggestion pill buttons */
    .stButton>button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s ease;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid rgba(148, 163, 184, 0.2);
        padding-bottom: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 600;
        font-size: 0.92rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- MODELS SETUP ----------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

@st.cache_resource
def load_llm():
    # Support both os.getenv and st.secrets (for Streamlit Cloud deployment)
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets.get("GOOGLE_API_KEY", None)
        except Exception:
            pass

    if api_key:
        return ChatGoogleGenerativeAI(
            model="gemini-3.5-flash",
            temperature=0,
            google_api_key=api_key
        )
    return ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        temperature=0
    )

embeddings = load_embeddings()
llm = load_llm()

def get_text(response):
    if isinstance(response.content, str):
        return response.content
    elif isinstance(response.content, list):
        text = ""
        for item in response.content:
            if isinstance(item, dict):
                text += item.get("text", "")
            elif hasattr(item, "text"):
                text += item.text
            else:
                text += str(item)
        return text
    return str(response.content)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_paper_names" not in st.session_state:
    st.session_state.uploaded_paper_names = []
if "total_chunks" not in st.session_state:
    st.session_state.total_chunks = 0
if "total_pages" not in st.session_state:
    st.session_state.total_pages = 0
if "active_tab" not in st.session_state:
    st.session_state.active_tab = 0

# ---------------- HERO HEADER ----------------
st.markdown(
    """
    <div class="hero-banner">
        <h1 class="hero-title">🩺 AI Healthcare Research Assistant</h1>
        <div class="hero-subtitle">
            Evidence-based clinical literature analysis, cross-paper synthesis, and RAG-powered querying.
        </div>
        <div class="hero-badges">
            <span class="hero-badge">⚡ Gemini Flash LLM</span>
            <span class="hero-badge">📚 FAISS Vector Engine</span>
            <span class="hero-badge">🔬 MiniLM Semantic Embeddings</span>
            <span class="hero-badge">📑 Multi-Document RAG</span>
            <span class="hero-badge">🔒 Private & Secure</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------- SIDEBAR: DOCUMENT & PIPELINE MANAGER ----------------
with st.sidebar:
    st.markdown("### 📚 Document Management")
    st.caption("Upload clinical trials, meta-analyses, or healthcare AI research papers (PDF).")

    uploaded_files = st.file_uploader(
        "Upload Research Papers (PDF)",
        type="pdf",
        accept_multiple_files=True,
        help="You can upload multiple medical papers simultaneously."
    )

    process_btn = st.button(
        "⚡ Process & Index Papers",
        type="primary",
        use_container_width=True,
        disabled=not uploaded_files
    )

    if process_btn and uploaded_files:
        all_documents = []
        progress_bar = st.progress(0, text="Initializing paper extraction...")

        with st.spinner("Extracting text and metadata from research papers..."):
            for idx, uploaded_file in enumerate(uploaded_files):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
                    f.write(uploaded_file.getvalue())
                    pdf_path = f.name

                loader = PyPDFLoader(pdf_path)
                documents = loader.load()

                for doc in documents:
                    doc.metadata["source"] = uploaded_file.name

                all_documents.extend(documents)
                progress_bar.progress(
                    int((idx + 1) / len(uploaded_files) * 60),
                    text=f"Loaded {idx + 1}/{len(uploaded_files)}: {uploaded_file.name[:25]}..."
                )

            progress_bar.progress(70, text="Chunking text with semantic overlap...")
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=300
            )
            chunks = splitter.split_documents(all_documents)

            progress_bar.progress(85, text="Computing dense embeddings in FAISS vector store...")
            db = FAISS.from_documents(chunks, embeddings)

            # Store in session state
            st.session_state["db"] = db
            st.session_state.uploaded_paper_names = [f.name for f in uploaded_files]
            st.session_state.total_chunks = len(chunks)
            st.session_state.total_pages = len(all_documents)

            progress_bar.progress(100, text="Vector indexing complete!")

        st.success(f"✅ Successfully indexed {len(uploaded_files)} papers ({len(chunks)} chunks)!")

    st.markdown("---")

    # Knowledge Base Status
    st.markdown("### 📊 Knowledge Base Status")
    if "db" in st.session_state:
        st.markdown(
            f"""
            <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid #10b981; border-radius: 8px; padding: 10px 14px; margin-bottom: 12px;">
                <span style="color: #059669; font-weight: 700; font-size: 0.88rem;">🟢 Vector Index Ready</span>
                <div style="font-size: 0.8rem; color: #4b5563; margin-top: 2px;">Ready for RAG Q&A, synthesis, and comparison.</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-num">{len(st.session_state.uploaded_paper_names)}</div>
                    <div class="metric-lbl">Papers</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with col_m2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-num">{st.session_state.total_chunks}</div>
                    <div class="metric-lbl">Text Chunks</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with st.expander("📑 View Indexed Documents", expanded=False):
            for i, name in enumerate(st.session_state.uploaded_paper_names):
                st.markdown(f"**{i+1}.** `{name}`")

        if st.button("🗑️ Clear Vector Index", use_container_width=True):
            del st.session_state["db"]
            st.session_state.uploaded_paper_names = []
            st.session_state.total_chunks = 0
            st.session_state.total_pages = 0
            st.rerun()
    else:
        st.markdown(
            """
            <div style="background: rgba(100, 116, 139, 0.08); border: 1px dashed #94a3b8; border-radius: 8px; padding: 14px; text-align: center; color: #64748b; font-size: 0.85rem;">
                ⚪ <b>No papers indexed yet</b><br>
                Upload one or more PDF research papers above and click <b>Process & Index</b> to enable RAG.
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("### ⚙️ Architecture Specs")
    st.markdown(
        """
        - **LLM**: Gemini Flash (T=0)
        - **Embedding**: `all-MiniLM-L6-v2`
        - **Chunking**: 1500 chars (overlap 300)
        - **Retrieval**: FAISS Dense Index
        """
    )

    st.markdown("---")
    st.caption(
        "⚕️ **Disclaimer**: This tool is designed strictly for scientific research and literature analysis. Not for clinical diagnosis or prescription."
    )


# ---------------- MAIN APPLICATION TABS ----------------
tabs = st.tabs([
    "💬 Research Q&A",
    "📑 Multi-Paper Summary",
    "⚖️ Cross-Paper Comparison",
    "🧠 Medical & AI Explainer",
    "🕒 Chat History"
])

is_db_ready = "db" in st.session_state
if is_db_ready:
    db = st.session_state["db"]


# ========================================================
# TAB 1: RESEARCH Q&A
# ========================================================
with tabs[0]:
    st.markdown("### 💬 Ask Questions from Research Papers")
    st.caption("Ask specific clinical or technical questions. The assistant retrieves exact passages and quotes page numbers.")

    if not is_db_ready:
        st.info("👈 **Upload and process research papers in the sidebar** to start asking questions using RAG.")

        with st.container(border=True):
            st.markdown("#### 💡 What you can ask once papers are indexed:")
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(
                    """
                    - **Methodology & Cohorts**: *"What were the inclusion criteria and sample size for the clinical cohort?"*
                    - **Model Architectures**: *"Which deep learning architectures were evaluated, and how were hyperparameters tuned?"*
                    """
                )
            with col_b:
                st.markdown(
                    """
                    - **Preprocessing Techniques**: *"What techniques were used for artifact removal, normalization, or class balancing?"*
                    - **Evaluation & Results**: *"What were the reported sensitivity, specificity, and AUROC scores?"*
                    """
                )
    else:
        # Suggested prompt chips
        st.markdown("**Quick Prompts:**")
        col_q1, col_q2, col_q3, col_q4 = st.columns(4)
        quick_query = None
        with col_q1:
            if st.button("🎯 Primary Objectives", use_container_width=True):
                quick_query = "What is the primary objective, clinical hypothesis, and problem formulation of this research?"
        with col_q2:
            if st.button("🧪 Datasets & Cohorts", use_container_width=True):
                quick_query = "What datasets, sample sizes, patient cohorts, and demographics were analyzed?"
        with col_q3:
            if st.button("⚙️ Preprocessing & Models", use_container_width=True):
                quick_query = "What data preprocessing methods and AI/ML model architectures were evaluated?"
        with col_q4:
            if st.button("📈 Key Findings & Limits", use_container_width=True):
                quick_query = "What are the key quantitative results, clinical conclusions, and reported limitations?"

        if "qa_input" not in st.session_state or quick_query:
            if quick_query:
                st.session_state["qa_input"] = quick_query

        with st.form(key="qa_form", clear_on_submit=False):
            question = st.text_input(
                "Enter your research question:",
                value=st.session_state.get("qa_input", ""),
                placeholder="e.g., What preprocessing techniques were applied before training the neural network?",
            )
            submit_qa = st.form_submit_button("🔍 Get Research Answer", type="primary", use_container_width=True)

        if submit_qa and question:
            with st.spinner("Searching indexed research papers and synthesizing answer..."):
                docs = db.similarity_search(question, k=6)
                context = "\n\n".join([d.page_content for d in docs])

                prompt = f"""
You are an AI Healthcare Research Assistant.

Answer only from the given research paper context.

Provide:
- Direct answer
- Technical explanation
- Preprocessing/model details

Context:
{context}

Question:
{question}
"""
                response = llm.invoke(prompt)
                answer = get_text(response)

                # Append to chat history
                st.session_state.messages.append({
                    "q": question,
                    "a": answer,
                    "sources": [
                        {
                            "source": doc.metadata.get("source", "Unknown"),
                            "page": doc.metadata.get("page", 0) + 1,
                            "content": doc.page_content[:280].strip()
                        }
                        for doc in docs
                    ]
                })

            st.markdown("#### 📋 Research Answer")
            st.markdown(
                f"""
                <div class="response-card">
                    {answer}
                </div>
                """,
                unsafe_allow_html=True
            )

            # Sources Section
            st.markdown("#### 📚 Retrieved Evidence & Sources")
            for i, doc in enumerate(docs):
                source_name = doc.metadata.get('source', 'Unknown Paper')
                page_num = doc.metadata.get('page', 0) + 1
                with st.expander(f"📄 Source {i+1}: {source_name} (Page {page_num})"):
                    st.write(doc.page_content)

            st.download_button(
                label="📥 Download Answer & Sources (Markdown)",
                data=f"# Research Q&A\n\n**Question:** {question}\n\n## Answer\n{answer}\n\n## Sources\n" + "\n".join([f"- **{d.metadata.get('source')}** (Page {d.metadata.get('page', 0)+1})\n> {d.page_content[:200]}..." for d in docs]),
                file_name="healthcare_research_qa.md",
                mime="text/markdown"
            )


# ========================================================
# TAB 2: MULTI-PAPER SUMMARY
# ========================================================
with tabs[1]:
    st.markdown("### 📑 Multi-Paper Comprehensive Summary")
    st.caption("Automatically synthesize the core objectives, datasets, preprocessing pipelines, model architectures, results, and limitations.")

    if not is_db_ready:
        st.info("👈 **Upload and process research papers in the sidebar** to enable comprehensive multi-paper summarization.")
    else:
        st.markdown(
            """
            This feature queries all indexed papers across 6 critical dimensions:
            - **1. Objective** • **2. Dataset** • **3. Preprocessing** • **4. Architecture** • **5. Results** • **6. Limitations**
            """
        )

        summarize_btn = st.button("📋 Generate Comprehensive Paper Summary", type="primary", use_container_width=True)

        if "last_summary" not in st.session_state:
            st.session_state.last_summary = None

        if summarize_btn:
            with st.spinner("Extracting research dimensions across indexed literature..."):
                docs = db.similarity_search(
                    "objective dataset preprocessing model results conclusion",
                    k=6
                )
                context = "\n\n".join([d.page_content for d in docs])

                prompt = f"""
Summarize these healthcare research papers.

Include:
1. Objective
2. Dataset
3. Preprocessing techniques
4. Model architecture
5. Results
6. Limitations

Context:
{context}
"""
                result = llm.invoke(prompt)
                st.session_state.last_summary = get_text(result)

        if st.session_state.last_summary:
            st.markdown("#### 📊 Synthesis Report")
            st.markdown(
                f"""
                <div class="response-card">
                    {st.session_state.last_summary}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.download_button(
                label="📥 Download Summary Report (Markdown)",
                data=f"# Healthcare Research Synthesis Report\n\n{st.session_state.last_summary}",
                file_name="research_papers_summary.md",
                mime="text/markdown"
            )


# ========================================================
# TAB 3: CROSS-PAPER COMPARISON
# ========================================================
with tabs[2]:
    st.markdown("### ⚖️ Cross-Paper Comparison")
    st.caption("Benchmark methodologies, datasets, models, performances, advantages, and limitations across papers.")

    if not is_db_ready:
        st.info("👈 **Upload and process research papers in the sidebar** to enable cross-paper comparative analysis.")
    else:
        st.markdown("**Example Comparison Topics:**")
        col_c1, col_c2, col_c3 = st.columns(3)
        quick_compare = None
        with col_c1:
            if st.button("📊 Model Accuracy & Metrics", use_container_width=True):
                quick_compare = "Compare the model architectures, validation protocols, and accuracy/AUROC results."
        with col_c2:
            if st.button("🧪 Datasets & Sample Sizes", use_container_width=True):
                quick_compare = "Compare the datasets, patient sample sizes, and demographic distributions."
        with col_c3:
            if st.button("⚙️ Preprocessing & Features", use_container_width=True):
                quick_compare = "Compare the data preprocessing techniques, data augmentation, and feature engineering."

        if "compare_input" not in st.session_state or quick_compare:
            if quick_compare:
                st.session_state["compare_input"] = quick_compare

        with st.form(key="compare_form"):
            compare = st.text_input(
                "Enter comparison query:",
                value=st.session_state.get("compare_input", ""),
                placeholder="e.g., Compare the machine learning models, validation strategies, and performance metrics across the papers.",
            )
            submit_compare = st.form_submit_button("⚖️ Run Comparative Analysis", type="primary", use_container_width=True)

        if "last_comparison" not in st.session_state:
            st.session_state.last_comparison = None

        if submit_compare and compare:
            with st.spinner("Retrieving relevant literature across papers and conducting comparative analysis..."):
                docs = db.similarity_search(compare, k=8)
                context = "\n\n".join([d.page_content for d in docs])

                prompt = f"""
Compare the research papers based on:

- Dataset
- Preprocessing
- Model architecture
- Performance
- Advantages
- Limitations

Context:
{context}

Question:
{compare}
"""
                result = llm.invoke(prompt)
                st.session_state.last_comparison = get_text(result)

        if st.session_state.last_comparison:
            st.markdown("#### ⚖️ Comparative Findings")
            st.markdown(
                f"""
                <div class="response-card">
                    {st.session_state.last_comparison}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.download_button(
                label="📥 Download Comparison (Markdown)",
                data=f"# Research Papers Comparison\n\n{st.session_state.last_comparison}",
                file_name="paper_comparison.md",
                mime="text/markdown"
            )


# ========================================================
# TAB 4: MEDICAL & AI TERM EXPLAINER
# ========================================================
with tabs[3]:
    st.markdown("### 🧠 Medical & AI Terminology Explainer")
    st.caption("Translate complex medical nomenclature and artificial intelligence concepts into clear, structured explanations.")

    st.markdown("**Popular Medical / AI Terms:**")
    col_t1, col_t2, col_t3, col_t4 = st.columns(4)
    quick_term = None
    with col_t1:
        if st.button("AUROC vs AUPRC", use_container_width=True):
            quick_term = "AUROC vs AUPRC in Imbalanced Clinical Data"
    with col_t2:
        if st.button("Attention Mechanism", use_container_width=True):
            quick_term = "Self-Attention Mechanism in Medical Imaging"
    with col_t3:
        if st.button("Federated Learning", use_container_width=True):
            quick_term = "Federated Learning in Healthcare"
    with col_t4:
        if st.button("Data Leakage", use_container_width=True):
            quick_term = "Data Leakage in Electronic Health Records"

    if "term_input" not in st.session_state or quick_term:
        if quick_term:
            st.session_state["term_input"] = quick_term

    with st.form(key="term_form"):
        term = st.text_input(
            "Enter medical or AI term:",
            value=st.session_state.get("term_input", ""),
            placeholder="e.g., Transformer architecture in ECG classification, AUC-ROC, DICOM format"
        )
        submit_term = st.form_submit_button("💡 Explain Term", type="primary", use_container_width=True)

    if "last_term_explanation" not in st.session_state:
        st.session_state.last_term_explanation = None

    if submit_term and term:
        with st.spinner("Generating structured explanation..."):
            prompt = f"""
Explain this term:

{term}

Give:
1. Simple explanation
2. Technical definition
3. Healthcare application
"""
            result = llm.invoke(prompt)
            st.session_state.last_term_explanation = get_text(result)

    if st.session_state.last_term_explanation:
        st.markdown("#### 💡 Explanation Breakdown")
        st.markdown(
            f"""
            <div class="response-card">
                {st.session_state.last_term_explanation}
            </div>
            """,
            unsafe_allow_html=True
        )


# ========================================================
# TAB 5: CHAT HISTORY
# ========================================================
with tabs[4]:
    st.markdown("### 🕒 Session Query History")
    st.caption("Review your questions and answers from this research session.")

    if not st.session_state.messages:
        st.markdown(
            """
            <div style="background: rgba(100, 116, 139, 0.05); border: 1px dashed #cbd5e1; border-radius: 8px; padding: 24px; text-align: center; color: #64748b;">
                💬 No questions asked in this session yet. Use the <b>Research Q&A</b> tab to start querying your papers!
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        col_h1, col_h2 = st.columns([4, 1])
        with col_h1:
            st.markdown(f"**Total Queries in Session:** `{len(st.session_state.messages)}`")
        with col_h2:
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

        st.markdown("---")

        for idx, chat in enumerate(reversed(st.session_state.messages)):
            actual_num = len(st.session_state.messages) - idx
            with st.container(border=True):
                st.markdown(f"**Q{actual_num}: {chat['q']}**")
                st.markdown(chat["a"])

                if "sources" in chat and chat["sources"]:
                    with st.expander(f"View Sources ({len(chat['sources'])})"):
                        for s_idx, src in enumerate(chat["sources"]):
                            st.markdown(f"- **{src['source']}** (Page {src['page']})")
                            st.caption(f"> {src['content']}...")

        # Export full transcript
        full_transcript = "# Research Session Q&A Transcript\n\n"
        for i, c in enumerate(st.session_state.messages):
            full_transcript += f"### Q{i+1}: {c['q']}\n\n**Answer:**\n{c['a']}\n\n---\n\n"

        st.download_button(
            label="📥 Export Full Chat Transcript (Markdown)",
            data=full_transcript,
            file_name="healthcare_research_transcript.md",
            mime="text/markdown"
        )
