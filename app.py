import streamlit as st
from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.embeddings.base import Embeddings

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSeq2SeqLM,
    pipeline
)

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Legal Document Analysis – RAG",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# FULL-SCREEN NEON BLACK THEME (WHITE TEXT)
# --------------------------------------------------
st.markdown("""
<style>

/* Base */
html, body, .stApp {
    background-color: #000000;
    color: #ffffff;
}

/* Remove default padding */
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 1.2rem;
}

/* Headings */
h1, h2, h3 {
    color: #ffffff;
    font-weight: 600;
    letter-spacing: 0.6px;
}

/* Divider */
.neon-divider {
    height: 2px;
    background: linear-gradient(90deg, transparent, #a855f7, transparent);
    margin: 24px 0;
}

/* Inputs */
input, textarea {
    background-color: #000000 !important;
    color: #ffffff !important;
    border: 1px solid #a855f7 !important;
    border-radius: 6px !important;
}

/* Labels */
label {
    color: #ffffff !important;
}

/* Buttons */
button {
    background-color: #000000 !important;
    color: #ffffff !important;
    border: 1px solid #a855f7 !important;
    border-radius: 6px !important;
}
button:hover {
    background-color: #a855f7 !important;
    color: #000000 !important;
}

/* Answer box */
.answer-box {
    border-left: 4px solid #a855f7;
    padding: 16px;
    background-color: #050505;
    color: #ffffff;
    font-size: 16px;
}

/* Source box */
.source-box {
    border: 1px solid #222222;
    padding: 10px;
    margin-bottom: 6px;
    background-color: #020202;
    color: #ffffff;
    font-size: 14px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #000000;
    border-right: 1px solid #1f1f1f;
}

/* Hide footer */
footer { visibility: hidden; }

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HEADER + IMAGE (HERO SECTION)
# --------------------------------------------------
st.markdown(
    "<h1 style='text-align:center;'>LEGAL DOCUMENT ANALYSIS & Q&A</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;'>Retrieval-Augmented Generation (RAG) Framework</p>",
    unsafe_allow_html=True
)
st.markdown("<div class='neon-divider'></div>", unsafe_allow_html=True)

# Hero image (dark compatible)
st.image(
    "https://images.unsplash.com/photo-1589829545856-d10d557cf95f",
    use_container_width=True
)

st.markdown("<div class='neon-divider'></div>", unsafe_allow_html=True)

# --------------------------------------------------
# EMBEDDINGS (STABLE FOR YOUR VERSIONS)
# --------------------------------------------------
class HFTransformerEmbeddings(Embeddings):
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def embed_documents(self, texts):
        vectors = []
        for text in texts:
            encoded = self.tokenizer(
                text,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt"
            )
            with torch.no_grad():
                output = self.model(**encoded)
            embedding = output.last_hidden_state.mean(dim=1)
            vectors.append(embedding.squeeze().cpu().numpy())
        return vectors

    def embed_query(self, text):
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
        with torch.no_grad():
            output = self.model(**encoded)
        return output.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.markdown("### Upload Legal Documents")
uploaded_files = st.sidebar.file_uploader(
    "PDF files only",
    type=["pdf"],
    accept_multiple_files=True
)

# --------------------------------------------------
# BUILD RAG PIPELINE
# --------------------------------------------------
@st.cache_resource(show_spinner=True)
def build_rag_pipeline(files):
    documents = []

    for file in files:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text()
        documents.append({"text": text, "source": file.name})

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    texts, metas = [], []
    for doc in documents:
        for chunk in splitter.split_text(doc["text"]):
            texts.append(chunk)
            metas.append({"source": doc["source"]})

    embeddings = HFTransformerEmbeddings()
    vectorstore = FAISS.from_texts(texts, embeddings, metas)

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512,
        temperature=0.0
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

# --------------------------------------------------
# MAIN INTERACTION
# --------------------------------------------------
if uploaded_files:
    qa_chain = build_rag_pipeline(uploaded_files)

    question = st.text_input("Ask a legal question")

    if question:
        with st.spinner("Analyzing legal documents..."):
            response = qa_chain(question)

        st.markdown("### Answer")
        st.markdown(
            f"<div class='answer-box'>{response['result']}</div>",
            unsafe_allow_html=True
        )

        st.markdown("### Source Documents")
        for doc in response["source_documents"]:
            st.markdown(
                f"<div class='source-box'>{doc.metadata['source']}</div>",
                unsafe_allow_html=True
            )
else:
    st.info("Upload legal PDF documents from the sidebar to begin.")
