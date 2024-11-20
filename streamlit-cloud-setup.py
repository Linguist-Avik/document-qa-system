# requirements.txt
transformers==4.37.2
torch==2.1.2
streamlit==1.31.0
PyPDF2==3.0.1
sentence-transformers==2.2.2

# .gitignore
__pycache__/
*.pyc
.env
venv/
.idea/
.streamlit/

# README.md
# Document QA System

A Streamlit-based Q&A system for PDF documents using Phi-3 model.

## Setup

1. Clone this repository
2. Install requirements: `pip install -r requirements.txt`
3. Run locally: `streamlit run streamlit_app.py`

## Usage

1. Upload a PDF document (max 10 pages)
2. Ask questions about the document content
3. View answers with source citations
