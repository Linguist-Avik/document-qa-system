from transformers import AutoModelForCausalLM, AutoTokenizer
import PyPDF2
import torch

class DocumentQASystem:
    def __init__(self):
        # Initialize the model and tokenizer (using Phi-3)
        try:
            self.model_name = "microsoft/Phi-3-mini-4k-instruct"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            print("Model and tokenizer loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")

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
        inputs = self.tokenizer.encode_plus(question, context, return_tensors="pt", max_length=512, truncation=True)
        
        # Run model inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        start_idx = outputs.logits.argmax()
        end_idx = outputs.logits.argmax()

        answer = self.tokenizer.decode(inputs["input_ids"][0][start_idx:end_idx + 1])
        source = context[max(0, inputs["input_ids"][0][start_idx - 20]) : end_idx + 20]
        return {"answer": answer, "source": source}
