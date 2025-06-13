import os
import gradio as gr
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_cohere import ChatCohere  #  Correct import for chat model

#  Ensure API key is set in environment
if not os.getenv("COHERE_API_KEY"):
    raise ValueError("COHERE_API_KEY environment variable is not set.")

#  Load embedding model
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#  Load FAISS vectorstore
vectorstore_path = os.path.abspath(
    r"C:\Users\solan\Documents\wasserstoff task\wasserstoff\chatbot_theme_identifier\backend\faiss_index"
)
db = FAISS.load_local(vectorstore_path, embedding, allow_dangerous_deserialization=True)

#  Set up LLM using ChatCohere (for chat models like 'command-xlarge-nightly')
llm = ChatCohere(model="command-xlarge-nightly", temperature=0)
retriever = db.as_retriever()
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

#  Define Gradio app
def answer_question(query):
    return qa_chain.run(query)

demo = gr.Interface(fn=answer_question, inputs="text", outputs="text", title="Chatbot Theme Identifier")

if __name__ == "__main__":
    demo.launch()
