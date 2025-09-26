# app.py
import os, re
import streamlit as st
from typing import List, Dict
from io import BytesIO
from fpdf import FPDF

from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document

from transformers import pipeline
import speech_recognition as sr
import pyttsx3
from PIL import Image
import pytesseract

import json

HISTORY_FILE = "chat_history.json"

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_history(history):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


# ----------------------
# Helpers
# ----------------------
def extract_pages_from_file(uploaded_file) -> List[Dict]:
    reader = PdfReader(uploaded_file)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if not text.strip():
            # OCR fallback for image PDFs
            try:
                x_object = page.get('/Resources', {}).get('/XObject', {})
                for obj in x_object:
                    if x_object[obj]['/Subtype'] == '/Image':
                        size = (x_object[obj]['/Width'], x_object[obj]['/Height'])
                        data = x_object[obj].get_data()
                        img = Image.frombytes("RGB", size, data)
                        text = pytesseract.image_to_string(img)
            except Exception:
                pass
        if text.strip():
            pages.append({"text": text.strip(), "meta": {"source": uploaded_file.name, "page": i + 1}})
    return pages

def create_documents_from_pages(pages: List[Dict], chunk_size=800, chunk_overlap=100) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs: List[Document] = []
    for p in pages:
        chunks = splitter.split_text(p["text"])
        for chunk in chunks:
            doc = Document(page_content=chunk, metadata={"source": p["meta"]["source"], "page": p["meta"]["page"]})
            docs.append(doc)
    return docs

def highlight_html(text: str, query: str) -> str:
    esc_q = re.escape(query.strip())
    if esc_q:
        pattern = re.compile(esc_q, re.IGNORECASE)
        text = pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", text)
    return text

# Persistent conversation history
if "history" not in st.session_state:
    st.session_state.history = load_history()

# ----------------------
# Streamlit UI
# ----------------------
st.set_page_config(page_title="DocuMind Ultimate", layout="wide")
st.title("📚 DocuMind Ultimate — Multi-PDF AI Q&A")

uploaded_files = st.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)
if not uploaded_files:
    st.info("Upload PDFs to enable Q&A.")
    st.stop()

# ----------------------
# Parse PDFs
# ----------------------
all_pages = []
for f in uploaded_files:
    pages = extract_pages_from_file(f)
    all_pages.extend(pages)

if not all_pages:
    st.error("No extractable text found in uploaded PDFs.")
    st.stop()

docs = create_documents_from_pages(all_pages, chunk_size=1000, chunk_overlap=200)
st.success(f"Processed {len(uploaded_files)} file(s), created {len(docs)} chunks.")

# ----------------------
# Embeddings + FAISS
# ----------------------
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
with st.spinner("Creating embeddings..."):
    embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectorstore = FAISS.from_documents(docs, embedder)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# ----------------------
# Summarization Mode
# ----------------------
st.subheader("📌 Options")
if st.button("Summarize whole PDF(s)"):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    full_text = " ".join([d.page_content for d in docs])[:6000]
    with st.spinner("Summarizing..."):
        summary = summarizer(full_text, max_length=400, min_length=80, do_sample=False)[0]['summary_text']
    st.info(summary)

# ----------------------
# Voice Q&A
# ----------------------
st.subheader("🎙️ Ask a Question")
use_mic = st.checkbox("Use Microphone")
if use_mic:
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening... Speak now")
        audio = recognizer.listen(source)
        try:
            query = recognizer.recognize_google(audio)
            st.success(f"You asked: {query}")
        except Exception as e:
            st.error(f"Speech recognition failed: {e}")
            query = ""
else:
    query = st.text_input("Type your question:")

if not query:
    st.stop()

# ----------------------
# Multi-answer retrieval
# ----------------------
with st.spinner("Retrieving relevant chunks..."):
    results = vectorstore.similarity_search_with_score(query, k=3)

if not results:
    st.write("No relevant results found.")
    st.stop()

st.subheader("🔎 Top Answers")
answer = ""
for i, (doc, _) in enumerate(results, start=1):
    snippet = doc.page_content
    answer += snippet + "\n"
    st.markdown(f"**{i}. {doc.metadata.get('source','?')} — Page {doc.metadata.get('page','?')}**")
    with st.expander("Show text"):
        st.markdown(highlight_html(snippet, query), unsafe_allow_html=True)

# ----------------------
# Text-to-Speech (TTS)
# ----------------------
if st.checkbox("🔊 Speak Answer"):
    try:
        engine = pyttsx3.init()
        engine.say(answer)
        engine.runAndWait()
    except Exception as e:
        st.error(f"TTS failed: {e}")

# ----------------------
# Save conversation history
# ----------------------
st.session_state.history.append({"Q": query, "A": answer})
save_history(st.session_state.history)

st.subheader("💾 Export Conversation")
export_type = st.radio("Export as:", ["Text", "PDF"])
if st.button("Download Conversation"):
    if export_type == "Text":
        output = "\n\n".join([f"Q: {h['Q']}\nA: {h['A']}" for h in st.session_state.history])
        st.download_button("Download TXT", output, file_name="conversation.txt")
    else:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        for h in st.session_state.history:
            pdf.multi_cell(0, 10, f"Q: {h['Q']}\nA: {h['A']}\n\n")
        buf = BytesIO()
        pdf_bytes = pdf.output(dest='S').encode('latin-1')
        st.download_button("Download PDF", data=pdf_bytes, file_name="conversation.pdf", mime="application/pdf")
