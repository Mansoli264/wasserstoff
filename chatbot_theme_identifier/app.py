# interface/app.py
import gradio as gr
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "backend"))
from chatbot import get_answers_and_themes

def chat_interface(question):
    answer, citations = get_answers_and_themes(question)

    table = "### Document Citations:\n| DocID | Page | Paragraph | Extracted Answer |\n|-------|------|-----------|------------------|\n"
    for row in citations:
        table += f"| {row['doc_id']} | {row['page']} | {row['paragraph']} | {row['text'][:50]}... |\n"

    return f"**Theme Summary:**\n{answer}\n\n{table}"

gr.Interface(fn=chat_interface, inputs="text", outputs="markdown", title="📚 Theme-Aware Chatbot").launch()
