import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_cohere import CohereEmbeddings
from langchain_community.chat_models import ChatCohere
from langchain.chains.question_answering import load_qa_chain

class KnowledgeBase:
    def __init__(self, store_dir="vectorstores/"):

        load_dotenv()  # Load environment variables from .env

        cohere_key = os.getenv("COHERE_API_KEY")
        if not cohere_key:
            raise ValueError("COHERE_API_KEY is not set in the environment.")

        # Pass model name explicitly to avoid errors
        self.embeddings = CohereEmbeddings(
            cohere_api_key=cohere_key,
            model="embed-english-v3.0"
        )

        self.store_dir = store_dir
        self.vectorstores = self.load_all_vectorstores()
        self.llm = ChatCohere(cohere_api_key=cohere_key)

        print(" KnowledgeBase initialized successfully.")
    def load_all_vectorstores(self):
        vectorstores = []

        if not os.path.exists(self.store_dir):
            os.makedirs(self.store_dir)

        for dir_name in os.listdir(self.store_dir):
            path = os.path.join(self.store_dir, dir_name)
            if os.path.isdir(path):
                try:
                    vs = FAISS.load_local(
                        path,
                        self.embeddings,
                        allow_dangerous_deserialization=True
                    )
                    vectorstores.append(vs)
                except Exception as e:
                    print(f"Could not load vectorstore at {path}: {e}")
        return vectorstores

    def query(self, question):
        all_docs = []
        for vs in self.vectorstores:
            docs = vs.similarity_search(question)
            all_docs.extend(docs)

        if not all_docs:
            return "No relevant documents found."

        qa_chain = load_qa_chain(self.llm, chain_type="stuff")
        return qa_chain.run(input_documents=all_docs, question=question)
