import os
from typing import Dict, Any
import librosa
from .base_reader import BaseDocumentReader

class AudioDocumentReader(BaseDocumentReader):
    def __init__(self, file_path: str):
        super().__init__(file_path)
        self.audio = None
        self.sr = None
    
    def read(self) -> tuple:
        if self.file_type in ['mp3', 'wav']:
            self.audio, self.sr = librosa.load(self.file_path)
        else:
            raise ValueError(f"Unsupported audio type: {self.file_type}")
        return self.audio, self.sr
    
    def extract_metadata(self) -> Dict[str, Any]:
        metadata = {
            'file_path': self.file_path,
            'file_type': self.file_type,
            'file_size': os.path.getsize(self.file_path),
            'file_name': os.path.basename(self.file_path),
        }
        if self.audio is not None and self.sr is not None:
            metadata['duration'] = len(self.audio) / self.sr
            metadata['sample_rate'] = self.sr
            metadata['num_samples'] = len(self.audio)
        self.metadata = metadata
        return metadata
    
    def get_content(self) -> tuple:
        if self.audio is None or self.sr is None:
            self.read()
        return self.audio, self.sr