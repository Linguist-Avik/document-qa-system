from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import List, Dict, Optional, Tuple
import numpy as np
from pathlib import Path
from dataclasses import dataclass
import PyPDF2
from sentence_transformers import SentenceTransformer
import textwrap

@dataclass
class DocumentChunk:
    text: str
    page_num: int
    chunk_num: int
    embedding: Optional[np.ndarray] = None

@dataclass
class Document:
    filename: str
    chunks: List[DocumentChunk]
    total_pages: int
    
class DocumentQASystem:
    def __init__(self, model_name="microsoft/phi-3", use_4bit=True):
        self.current_document: Optional[Document] = None
        self.chunk_size = 512
        self.overlap = 50
        self.context_window = []  # Store recent Q&A pairs
        self.max_context_pairs = 3  # Number of Q&A pairs to maintain in context
        
        # Initialize Phi-3
        config = {
            "load_in_4bit": use_4bit,
            "bnb_4bit_compute_dtype": torch.float16,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
        } if use_4bit else {"torch_dtype": torch.float16}
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True,
            **config
        )
        
        # Initialize embedding model for semantic search
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def process_pdf(self, pdf_path: str, filename: str) -> Tuple[bool, str]:
        """Process a PDF file and store its chunks"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Check page limit
                if len(pdf_reader.pages) > 10:
                    return False, "PDF exceeds 10 page limit"
                
                chunks = []
                chunk_num = 0
                
                # Process each page
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    
                    # Split page text into chunks
                    page_chunks = self._chunk_text(text)
                    
                    # Create DocumentChunk objects with embeddings
                    for text_chunk in page_chunks:
                        embedding = self.embedding_model.encode(text_chunk)
                        chunks.append(DocumentChunk(
                            text=text_chunk,
                            page_num=page_num + 1,
                            chunk_num=chunk_num,
                            embedding=embedding
                        ))
                        chunk_num += 1
                
                # Store as current document
                self.current_document = Document(
                    filename=filename,
                    chunks=chunks,
                    total_pages=len(pdf_reader.pages)
                )
                
                return True, f"Successfully processed {filename}"
                
        except Exception as e:
            return False, f"Error processing PDF: {str(e)}"
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        chunks = textwrap.wrap(text, self.chunk_size, break_long_words=True)
        
        overlapped_chunks = []
        for i, chunk in enumerate(chunks):
            if i > 0:
                chunk = chunks[i-1][-self.overlap:] + chunk
            if i < len(chunks) - 1:
                chunk = chunk + chunks[i+1][:self.overlap]
            overlapped_chunks.append(chunk)
            
        return overlapped_chunks
    
    def _get_relevant_chunks(self, question: str, n_chunks: int = 3) -> List[Tuple[DocumentChunk, float]]:
        """Get most relevant chunks using semantic search"""
        if not self.current_document:
            return []
        
        # Get question embedding
        question_embedding = self.embedding_model.encode(question)
        
        # Calculate similarities
        similarities = []
        for chunk in self.current_document.chunks:
            similarity = np.dot(question_embedding, chunk.embedding) / (
                np.linalg.norm(question_embedding) * np.linalg.norm(chunk.embedding)
            )
            similarities.append((chunk, similarity))
        
        # Sort by similarity and return top n
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:n_chunks]
    
    def answer_question(self, question: str) -> Tuple[str, List[Dict]]:
        """Generate answer for a question using relevant document chunks"""
        if not self.current_document:
            return "Please upload a document first.", []
        
        relevant_chunks = self._get_relevant_chunks(question)
        
        if not relevant_chunks:
            return "I couldn't find relevant information in the document to answer your question.", []
        
        # Create context from previous Q&A pairs
        context_text = ""
        if self.context_window:
            context_text = "Previous conversation:\n"
            for qa in self.context_window:
                context_text += f"Q: {qa['question']}\nA: {qa['answer']}\n\n"
        
        # Create prompt with context and relevant chunks
        prompt = f"""{context_text}
Based on the following excerpts from the document, answer the question.

Relevant excerpts:
{' '.join([chunk.text for chunk, _ in relevant_chunks])}

Question: {question}

Answer:"""
        
        # Generate answer
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_length=1024,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
        )
        
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Update context window
        self.context_window.append({'question': question, 'answer': answer})
        if len(self.context_window) > self.max_context_pairs:
            self.context_window.pop(0)
        
        # Prepare source citations
        sources = [
            {
                'page': chunk.page_num,
                'text': chunk.text[:200] + "...",  # Preview of the chunk
                'similarity': similarity
            }
            for chunk, similarity in relevant_chunks
        ]
        
        return answer, sources
    
    def clear_document(self):
        """Clear the current document and context"""
        self.current_document = None
        self.context_window = []
