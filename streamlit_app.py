import streamlit as st
import os
from pathlib import Path

# Importing the backend module
from doc_qa_backend import DocumentQASystem

class DocumentQAInterface:
    def __init__(self):
        self.initialize_session_state()
        self.setup_ui()

    def initialize_session_state(self):
        """Initialize session state variables."""
        if 'qa_system' not in st.session_state:
            st.session_state.qa_system = DocumentQASystem()

        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

    def setup_ui(self):
        """Setup Streamlit UI."""
        st.title("Document QA System")
        uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

        if uploaded_file:
            temp_file_path = self.save_uploaded_file(uploaded_file)
            document = st.session_state.qa_system.process_document(temp_file_path)
            st.write(f"Processed document: {document.filename}")
            st.write(f"Total pages: {document.total_pages}")

            for chunk in document.chunks:
                st.write(f"Page {chunk.page_num}: {chunk.text}")

    @staticmethod
    def save_uploaded_file(uploaded_file) -> str:
        """Save uploaded file to a temporary location."""
        temp_dir = Path(tempfile.gettempdir())
        temp_file_path = temp_dir / uploaded_file.name
        with open(temp_file_path, 'wb') as f:
            f.write(uploaded_file.read())
        return str(temp_file_path)

if __name__ == "__main__":
    app = DocumentQAInterface()
