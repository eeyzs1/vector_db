import os
from typing import Dict, Any
from PIL import Image
import numpy as np
from .base_reader import BaseDocumentReader

class ImageDocumentReader(BaseDocumentReader):
    def __init__(self, file_path: str):
        super().__init__(file_path)
        self.image = None
    
    def read(self) -> Image.Image:
        if self.file_type in ['jpg', 'jpeg', 'png', 'bmp']:
            self.image = Image.open(self.file_path)
        else:
            raise ValueError(f"Unsupported image type: {self.file_type}")
        return self.image
    
    def extract_metadata(self) -> Dict[str, Any]:
        metadata = {
            'file_path': self.file_path,
            'file_type': self.file_type,
            'file_size': os.path.getsize(self.file_path),
            'file_name': os.path.basename(self.file_path),
        }
        if self.image:
            metadata['width'], metadata['height'] = self.image.size
            metadata['mode'] = self.image.mode
        self.metadata = metadata
        return metadata
    
    def get_content(self) -> Image.Image:
        if not self.image:
            self.read()
        return self.image