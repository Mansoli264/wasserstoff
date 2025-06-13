Chatbot Theme Identifier  
This is a document-aware chatbot that processes your uploaded PDFs, scans, and text files to answer your questions with precise citations (page, paragraph, sentence). Moreover, it determines themes across documents and summarizes them in a conversational manner.  
This tool is tailored to meet the needs of legal practitioners, analysts, scholars, and other professionals who deal with extensive volumes of documents and require meaningful extractions.
 Overview
 Upload & Analyze Documents
Able to upload over 75 documents including pdfs, scanned files, DOCX, and TXT documents.
Has a feature where text is automatically extracted with OCR for scanned images.
All data is stored in a fast, searchable vector database (FAISS).
 Ask Questions in Natural Language
Available for use is a fully customizable question prompt, for example:  
"How do you handle overfitting in a machine learning model?"  
"How can you address the vanishing gradient problem in deep neural networks?"  
The bot conducts a comprehensive search across all documents and retrieves answers from the specific page and paragraph.
 Detailed Citations
For each response provided, the system generates detailed citations which include:  
 Document ID  
 Page number  
 Paragraph number  
 Thematic Analysis and Summarization  
Summarization identifies critical recurring elements and systematically organizes them into predefined labeled categories (e.g., Theme 1 – Regulatory Non-Compliance).
 User-Friendly Interface
Built using Gradio


Three tabs:


Ask Questions: Chat with the bot


Upload Docs: Add new documents to the knowledge base


View Docs: See what documents have been uploaded



 How It Works (Plain English)
You upload documents


They can be PDFs, scanned images (like printouts), Word files, or plain text


Text is extracted


Regular text is pulled out directly


Scanned images are processed using OCR (Optical Character Recognition)


Chunks are created


Each page is split into paragraphs


Each paragraph is stored with metadata: which file, which page, which paragraph


All text is indexed using AI embeddings


This means the bot can "understand" the meaning of the text


You ask a question


The chatbot finds the most relevant pieces from all documents


Then it uses a powerful language model (Cohere) to write a full answer


Themes are detected


The bot reads through all individual answers and groups them


You get a friendly summary with headings like "Theme 1 - Delay in Disclosure"



 Getting Started
Prerequisites
Python 3.10+


pip installed


API keys for:


Cohere


Installation
# Clone the repo
git clone https://github.com/your-username/chatbot-theme-identifier.git
cd chatbot-theme-identifier

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

Set Your API Key
# In your terminal, set the COHERE API Key (replace with your actual key)
$env:COHERE_API_KEY="your-api-key"  # For Windows
export COHERE_API_KEY="your-api-key"  # For Mac/Linux

Run the App
python gradio_app.py

Then open http://127.0.0.1:7860 in your browser.

🗂 Project Structure
chatbot-theme-identifier/
├── app/
│   └── api/
│       └── gradio_app.py       # Main chatbot app with Gradio UI
├── backend/
│   ├── ingest.py               # Handles document loading, OCR, and indexing
│   ├── faiss_index/            # Stores the semantic index for search
│   └── uploaded_docs/          # Folder to hold uploaded files
├── requirements.txt
├── README.md                   # This file


 Example Output
 Individual Document Answers
Document ID
Extracted Answer
Citation
DOC001
The fine was imposed under section 15...
Page 4, Para 2
DOC002
Delay in disclosure violated Clause 49...
Page 2, Para 1

 Synthesized Theme Answer
Theme 1 – Regulatory Non-Compliance:
DOC001, DOC002: Violation of SEBI Act and LODR.

Theme 2 – Penalty Justification:
DOC001: Clear statutory reasoning for penalties.


 FAQs
1. Can I upload scanned PDFs?
 Yes! The tool uses OCR to read text from scanned documents and images.
2. Will it work for 100+ documents?
 Yes, FAISS makes the search efficient even with hundreds of documents.
3. Can I get sentence-level citations?
 Yes, it includes metadata at the paragraph and sentence level.
4. What models does it use?
Embeddings: HuggingFace Sentence Transformers


Chat + Themes: Cohere Chat Model (e.g. command-r-plus)



 Tech Stack
Frontend: Gradio


Backend: Python (LangChain + Cohere)


Search Engine: FAISS


Embeddings: HuggingFace Transformers


OCR: Tesseract (optional, used in ingest.py)



 Optional Extras
Sentence-level citation links (clickable)


Document filter/sorting by date or type


Toggle documents on/off per query



Demo Video (Recommended)
Include a short video showing:
Uploading files

Asking a question
Viewing theme summaries
MIT License
Made with ❤️ by Manali Solanki for the Wasserstoff Challenge.

