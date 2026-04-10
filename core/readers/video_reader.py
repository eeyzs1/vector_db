import os
import cv2
from typing import Dict, Any, List
from .base_reader import BaseDocumentReader

class VideoDocumentReader(BaseDocumentReader):
    def __init__(self, file_path: str):
        super().__init__(file_path)
        self.cap = None
        self.frames = []
    
    def read(self) -> List[cv2.Mat]:
        if self.file_type in ['mp4', 'avi', 'mov']:
            self.cap = cv2.VideoCapture(self.file_path)
            if not self.cap.isOpened():
                raise ValueError(f"Could not open video file: {self.file_path}")
            
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                self.frames.append(frame)
            
            self.cap.release()
        else:
            raise ValueError(f"Unsupported video type: {self.file_type}")
        return self.frames
    
    def extract_metadata(self) -> Dict[str, Any]:
        metadata = {
            'file_path': self.file_path,
            'file_type': self.file_type,
            'file_size': os.path.getsize(self.file_path),
            'file_name': os.path.basename(self.file_path),
        }
        if self.cap:
            metadata['fps'] = self.cap.get(cv2.CAP_PROP_FPS)
            metadata['total_frames'] = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            metadata['width'] = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            metadata['height'] = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        elif self.frames:
            metadata['total_frames'] = len(self.frames)
            if self.frames:
                metadata['height'], metadata['width'] = self.frames[0].shape[:2]
        self.metadata = metadata
        return metadata
    
    def get_content(self) -> List[cv2.Mat]:
        if not self.frames:
            self.read()
        return self.frames