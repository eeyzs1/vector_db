import time
from typing import List, Tuple
import numpy as np
from .base_processor import BaseProcessor
from core.model_manager import model_manager


class AudioProcessor(BaseProcessor):
    def __init__(self, test_mode=False):
        self.test_mode = test_mode
        self.n_mfcc = 128
        self.segment_duration = 30  # seconds per chunk

    def chunk(self, content: Tuple) -> List[Tuple]:
        """Split audio into fixed-duration segments."""
        audio, sr = content
        segment_samples = self.segment_duration * sr
        if len(audio) <= segment_samples:
            return [(audio, sr)]
        segments = []
        for start in range(0, len(audio), segment_samples):
            segment = audio[start:start + segment_samples]
            if len(segment) > 0:
                segments.append((segment, sr))
        return segments

    def clean(self, content: Tuple) -> Tuple:
        """Normalize audio amplitude to [-1, 1]."""
        audio, sr = content
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
        return (audio, sr)

    def embed(self, content: Tuple) -> List[float]:
        start_time = time.time()

        if self.test_mode:
            import random
            embedding = [random.random() for _ in range(self.n_mfcc)]
        else:
            import librosa
            audio, sr = content
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=self.n_mfcc)
            embedding = mfcc.mean(axis=1).tolist()

        processing_time = time.time() - start_time
        model_manager.record_model_usage('audio_mfcc', 'audio', processing_time)
        return embedding
