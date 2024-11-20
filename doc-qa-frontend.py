import streamlit as st
import tempfile
from pathlib import Path
import os

class DocumentQAInterface:
    def __init__(self):
        self.initialize_session_state()
        self.setup_ui()
    
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'qa_system' not in st.session_state:
            from doc_qa_backend import DocumentQASystem
            st.session_state.qa_system = DocumentQASystem()
        
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
            
        if 'current_document' not in st.session_state:
            st.session_state.current_document = None
    
    def setup_ui(self):
        """Setup the Streamlit UI"""
        st.title("Document Q&A System")
        
        # Sidebar for document upload and info
        with st.sidebar:
            st.header("Document Upload")
            uploaded_file = st.file_uploader(
                "Upload a PDF document (max 10 pages)",
                type=['pdf'],
                accept_multiple_files=False
            )
            
            if uploaded_file:
                self.process_uploaded_file(uploaded_file)
            
            # Display current document info
            if st.session_state.qa_system.current_document:
                st.header("Current Document")
                doc = st.session_state.qa_system.current_document
                st.write(f"📄 {doc.filename}")
                st.write(f"Pages: {doc.total_pages}")
                
                if st.button("Clear Document"):
                    self.clear_current_document()
        
        # Main chat interface
        if not st.session_state.qa_system.current_document:
            st.info("Please upload a PDF document to start asking questions.")
            return
        
        st.header("Ask Questions")
        
        # Display chat history with sources
        for msg in st.session_state.chat_history:
            role = msg["role"]
            content = msg["content"]
            
            if role == "user":
                st.write(f"You: {content}")
            else:
                st.write(f"Assistant: {content}")
                
                # Display sources if available
                if "sources" in msg:
                    with st.expander("View Sources"):
                        for idx, source in enumerate(msg["sources"], 1):
                            st.markdown(f"""
                            **Source {idx}** (Page {source['page']}, Relevance: {source['similarity']:.2f})
                            ```
                            {source['text']}
                            ```
                            """)
        
        # Question input
        question = st.text_input("Type your question here")
        if st.button("Ask") and question:
            self.process_question(question)
    
    def process_uploaded_file(self, file):
        """Process an uploaded PDF file"""
        # Clear existing document if any
        if st.session_state.qa_system.current_document:
            self.clear_current_document()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(file.read())
            filepath = tmp_file.name
        
        try:
            success, message = st.session_state.qa_system.process_pdf(filepath, file.name)
            
            if success:
                st.session_state.current_document = file.name
                st.sidebar.success(message)
            else:
                st.sidebar.error(message)
            
        except Exception as e:
            st.sidebar.error(f"Error processing PDF: {str(e)}")
        
        finally:
            os.unlink(filepath)  # Clean up temp file
    
    def process_question(self, question: str):
        """Process a user question"""
        # Add user question to history
        st.session_state.chat_history.append({
            "role": "user",
            "content": question
        })
        
        # Get answer and sources
        with st.spinner("Thinking..."):
            answer, sources = st.session_state.qa_system.answer_question(question)
        
        # Add response to history
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })
        
        # Rerun to update display
        st.experimental_rerun()
    
    def clear_current_document(self):
        """Clear the current document and related state"""
        st.session_state.qa_system.clear_document()
        st.session_state.current_document = None
        st.session_state.chat_history = []
        st.experimental_rerun()

if __name__ == "__main__":
    st.set_page_config(
        page_title="Document Q&A System",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    app = DocumentQAInterface()
