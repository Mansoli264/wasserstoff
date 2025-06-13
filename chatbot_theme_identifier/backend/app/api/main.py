from dotenv import load_dotenv
import os
import logging
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Load environment
load_dotenv(dotenv_path=".env")  # Adjusted for root execution

# Initialize logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("wasserstoff")

# Initialize FastAPI app
app = FastAPI()

# Set max file size: 10 MB
MAX_FILE_SIZE = 10 * 1024 * 1024

@app.middleware("http")
async def limit_upload_size(request: Request, call_next):
    if request.headers.get("content-length"):
        if int(request.headers["content-length"]) > MAX_FILE_SIZE:
            logger.warning("Upload rejected: file too large")
            raise HTTPException(status_code=413, detail="File too large")
    return await call_next(request)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# LangChain and model imports
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Local module imports
from app.api.extractor import extract_text
from app.api.preprocessor import clean_text
from app.api.theme_analyzer import detect_themes
from app.api.knowledge_base import KnowledgeBase

# Models
class AskRequest(BaseModel):
    question: str
    doc_id: str

class ChatRequest(BaseModel):
    question: str

# Initialize knowledge base
temp_kb = None

@app.on_event("startup")
def startup_event():
    global temp_kb
    temp_kb = KnowledgeBase()

@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        os.makedirs("uploaded", exist_ok=True)
        os.makedirs("extracted", exist_ok=True)
        os.makedirs("vectorstores", exist_ok=True)

        file_location = f"uploaded/{file.filename}"
        with open(file_location, "wb") as f:
            f.write(await file.read())
        logger.info(f"File uploaded to {file_location}")

        extracted_text = extract_text(file_location)
        cleaned_text = clean_text(extracted_text)

        output_path = f"extracted/{file.filename}.txt"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(cleaned_text)

        doc_id = os.path.splitext(file.filename)[0]
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(cleaned_text)
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_texts(chunks, embeddings)
        vectorstore.save_local(f"vectorstores/{doc_id}")

        themes = detect_themes(cleaned_text)

        return {
            "message": "File uploaded, processed, and analyzed!",
            "filename": file.filename,
            "text_preview": cleaned_text[:300],
            "themes_detected": themes
        }

    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")

@app.post("/ask/")
async def ask_question(payload: AskRequest):
    question = payload.question
    doc_id = payload.doc_id
    vectorstore_path = f"vectorstores/{doc_id}"

    if not os.path.exists(vectorstore_path):
        raise HTTPException(status_code=404, detail="Document not found")

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)
    llm = ChatOpenAI()
    qa_chain = load_qa_chain(llm, chain_type="stuff")

    docs = vectorstore.similarity_search(question)
    if not docs:
        return {"answer": "No relevant content found in this document."}

    answer = qa_chain.run(input_documents=docs, question=question)
    return {"answer": answer}

@app.post("/chat/")
async def chat_across_all_docs(payload: ChatRequest):
    question = payload.question
    all_docs = []
    embeddings = OpenAIEmbeddings()
    vector_dir = "vectorstores"

    if not os.path.exists(vector_dir):
        raise HTTPException(status_code=404, detail="No documents uploaded yet.")

    for doc_id in os.listdir(vector_dir):
        doc_path = os.path.join(vector_dir, doc_id)
        if os.path.isdir(doc_path):
            try:
                store = FAISS.load_local(doc_path, embeddings, allow_dangerous_deserialization=True)
                docs = store.similarity_search(question, k=2)
                all_docs.extend(docs)
            except Exception as e:
                logger.warning(f"Could not load index {doc_id}: {e}")

    if not all_docs:
        return {"answer": "No relevant content found in uploaded documents."}

    llm = ChatOpenAI()
    chain = load_qa_chain(llm, chain_type="stuff")
    answer = chain.run(input_documents=all_docs, question=question)
    return {"answer": answer}

@app.post("/ask_all/")
async def ask_all_docs(payload: ChatRequest):
    answer = temp_kb.query(payload.question)
    return {"answer": answer}

# Debug log to confirm server start
print("API Routes:", [route.path for route in app.routes])
