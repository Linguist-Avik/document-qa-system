from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import PyPDF2
import os

class DocumentQASystem:
    def __init__(self):
        # Initialize the Phi-3 model and tokenizer
        self.model_name = "microsoft/Phi-3-mini-4k-instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.qa_pipeline = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)
        self.documents = {}

    def preprocess_pdf(self, pdf_path):
        """Extract text from a PDF file."""
        text = ""
        with open(pdf_path, "rb") as pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)
            for page in reader.pages:
                text += page.extract_text()
        return text

    def add_document(self, doc_id, pdf_path):
        """Add a document to the system."""
        text = self.preprocess_pdf(pdf_path)
        self.documents[doc_id] = text

    def answer_question(self, doc_id, question):
        """Answer a question based on a specific document."""
        if doc_id not in self.documents:
            return {"answer": "Document not found", "source": ""}

        context = self.documents[doc_id]
        
        # Prepare the input for the Phi-3 model
        input_text = f"Context: {context}\nQuestion: {question}\nAnswer:"

        # Use the Phi-3 model to generate an answer
        result = self.qa_pipeline(input_text, max_length=512, num_return_sequences=1)[0]

        # Extract the generated answer
        answer = result['generated_text'].strip().split("Answer:")[-1].strip()

        # Get a source snippet (for simplicity, just take a snippet of the context)
        source_start = max(0, input_text.find(question) - 50)  # 50 chars before the question
        source_end = min(len(context), input_text.find(question) + 200)  # 200 chars after the question
        source = context[source_start:source_end]

        return {"answer": answer, "source": source}

# Create an instance of the system
doc_qa_system = DocumentQASystem()

def answer_question_from_pdf(pdf_path, question):
    """Answer a question based on the uploaded PDF."""
    # Add the document to the system (with a unique doc_id)
    doc_id = os.path.basename(pdf_path)
    doc_qa_system.add_document(doc_id, pdf_path)

    # Get the answer for the question
    return doc_qa_system.answer_question(doc_id, question)
