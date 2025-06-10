
from app.services.embedding import search_similar_chunks
import openai
import os

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

def process_query(question: str):
    # 🔍 Step 1: Get the most relevant chunks from the vector DB
    context_chunks = search_similar_chunks(question)

    # 🧠 Step 2: Combine all context into one string
    context = "\n\n".join(context_chunks)

    # 💬 Step 3: Formulate the prompt to LLM
    prompt = f"""
You are a research assistant analyzing multiple documents.

Context from documents:
-----------------------
{context}

User Question:
--------------
{question}

Instructions:
1. Answer the question based on the context.
2. For each answer, cite the document ID, page number, and paragraph if mentioned.
3. Identify key themes across documents related to the question.
4. Output a summary with themes and related document IDs.

Now respond.
"""
 # 🤖 Step 4: Call OpenAI LLM (GPT-3.5-turbo)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    # 🧾 Step 5: Return the final structured response
    return {
        "question": question,
        "response": response.choices[0].message["content"],
        "context_used": context_chunks
    }