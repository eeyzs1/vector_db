import os
from typing import Dict, Any
import pdfplumber
from docx import Document
from .base_reader import BaseDocumentReader

class TextDocumentReader(BaseDocumentReader):
    def __init__(self, file_path: str):
        super().__init__(file_path)
        self.content = None
    
    def read(self) -> str:
        if self.file_type == 'txt' or self.file_type == 'md':
            with open(self.file_path, 'r', encoding='utf-8') as f:
                self.content = f.read()
        elif self.file_type == 'docx':
            doc = Document(self.file_path)
            self.content = '\n'.join([para.text for para in doc.paragraphs])
        elif self.file_type == 'pdf':
            with pdfplumber.open(self.file_path) as pdf:
                text = []
                for page in pdf.pages:
                    text.append(page.extract_text())
                self.content = '\n'.join(text)
        else:
            raise ValueError(f"Unsupported file type: {self.file_type}")
        return self.content
    
    def extract_metadata(self) -> Dict[str, Any]:
        metadata = {
            'file_path': self.file_path,
            'file_type': self.file_type,
            'file_size': os.path.getsize(self.file_path),
            'file_name': os.path.basename(self.file_path),
        }
        if self.content:
            metadata['content_length'] = len(self.content)
            metadata['word_count'] = len(self.content.split())
        self.metadata = metadata
        return metadata
    
    def get_content(self) -> str:
        if not self.content:
            self.read()
        return self.content