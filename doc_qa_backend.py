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
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        # Initialize other attributes here if needed.

    def process_document(self, filepath: str) -> Document:
        """Process a PDF file and return a Document object."""
        with open(filepath, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            chunks = []
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                chunks.append(DocumentChunk(text=text, page_num=i, chunk_num=i))
            return Document(filename=filepath, chunks=chunks, total_pages=len(reader.pages))
