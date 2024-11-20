import streamlit as st
from backend import DocumentQASystem
import os

# Initialize the backend system
qa_system = DocumentQASystem()

# Function to handle document upload
def upload_document():
    uploaded_file = st.file_uploader("Choose a PDF document", type="pdf")
    if uploaded_file is not None:
        # Save the uploaded file to the server
        file_path = os.path.join("uploaded_docs", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Add the document to the system
        doc_id = uploaded_file.name
        qa_system.add_document(doc_id, file_path)
        st.success(f"Document '{uploaded_file.name}' uploaded successfully!")

# Function to display the chat interface
def chat_interface():
    st.header("Ask Questions About Your Document")

    # Dropdown to select document
    doc_ids = list(qa_system.documents.keys())
    selected_doc = st.selectbox("Select a document", doc_ids)

    if selected_doc:
        question = st.text_input("Ask a question:")
        
        if st.button("Get Answer"):
            if question:
                response = qa_system.answer_question(selected_doc, question)
                st.write(f"**Answer:** {response['answer']}")
                st.write(f"**Source:** {response['source']}")
            else:
                st.warning("Please enter a question.")

# Main interface
def main():
    st.title("Document Question-Answering System")
    
    # Sidebar with file upload
    st.sidebar.header("Upload Document")
    upload_document()
    
    # Chat interface to ask questions
    chat_interface()

if __name__ == "__main__":
    # Ensure the uploaded documents folder exists
    if not os.path.exists("uploaded_docs"):
        os.makedirs("uploaded_docs")

    main()
