import os
import gradio as gr
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_cohere import ChatCohere
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ========== Configuration ==========
if not os.getenv("COHERE_API_KEY"):
    raise ValueError("COHERE_API_KEY environment variable is not set.")

#  Path to documents relative to this file
DOCS_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../docs/tests")
)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ========== Load Documents ==========
docs = []

for filename in os.listdir(DOCS_PATH):
    file_path = os.path.join(DOCS_PATH, filename)
    if filename.endswith(".txt"):
        loader = TextLoader(file_path)
        docs.extend(loader.load())
    elif filename.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
        docs.extend(loader.load())

if not docs:
    raise ValueError("No documents found in the 'docs' folder. Add .txt or .pdf files.")

# ========== Embed & Create Vector Store ==========
embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_docs = text_splitter.split_documents(docs)

db = FAISS.from_documents(split_docs, embedding)

# ========== LLM + QA Chain ==========
llm = ChatCohere(model="command-light", temperature=0)
retriever = db.as_retriever()
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# ========== Gradio Interface ==========
def answer_question(query):
    return qa_chain.run(query)

demo = gr.Interface(
    fn=answer_question,
    inputs="text",
    outputs="text",
    title="Chatbot Theme Identifier"
)

#  Launch with server_name and server_port for Render
if __name__ == "__main__":
    PORT = int(os.environ.get('PORT', 7860))
    demo.launch(server_name="0.0.0.0", server_port=PORT)

