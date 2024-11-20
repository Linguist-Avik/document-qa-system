# Document Question-Answering System

This project implements a document analysis system that allows users to upload PDF files and ask questions about the contents of the documents. The system uses a general-purpose question-answering model from Hugging Face and is built using Python and Streamlit.

## Features

- Upload PDF files through a user-friendly Streamlit interface.
- Ask questions based on the contents of uploaded documents.
- Display answers with excerpts from the document to provide context.

## Requirements

This project requires the following Python libraries:

- `streamlit`
- `transformers`
- `torch`
- `PyPDF2`

You can install them using the following command:

```bash
pip install -r requirements.txt
