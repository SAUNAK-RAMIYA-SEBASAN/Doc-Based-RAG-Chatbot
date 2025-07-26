import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Tuple
import os
from dataclasses import dataclass
import psutil
from tqdm import tqdm

@dataclass
class EmbeddingResult:
    embeddings: np.ndarray
    chunk_ids: List[str]
    metadata: List[Dict]

class EmbeddingEngine:
    def __init__(self, model_name: str = "intfloat/e5-small-v2", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.embedding_dim = 384  
        
    def load_model(self):
        """Load the embedding model with aggressive memory optimization"""
        print(f"Loading embedding model: {self.model_name}")
        
        memory_gb = psutil.virtual_memory().available / (1024**3)
        if memory_gb < 2.0:  
            raise RuntimeError(f"Insufficient memory: {memory_gb:.1f}GB available, need at least 2GB")
        
        self.model = SentenceTransformer(
            self.model_name, 
            device=self.device,
            cache_folder=None  
        )
        
        self.model.eval()
        
        print(f"✅ Model loaded successfully on {self.device}")

        
    def embed_chunks(self, chunks: List, batch_size: int = 32) -> EmbeddingResult:
        """Convert document chunks to embeddings"""
        if self.model is None:
            self.load_model()
        
        texts = []
        chunk_ids = []
        metadata = []
        
        for chunk in chunks:
            query_text = f"passage: {chunk.content}"
            texts.append(query_text)
            chunk_ids.append(chunk.chunk_id)
            metadata.append({
                'filename': chunk.metadata.get('filename', ''),
                'page_number': chunk.page_number,
                'section_name': chunk.section_name,
                'chunk_id': chunk.chunk_id
            })
        
        print(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Creating embeddings"):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(
                batch_texts,
                convert_to_tensor=False,
                normalize_embeddings=True,
                batch_size=len(batch_texts)
            )
            embeddings.extend(batch_embeddings)
        
        embeddings_array = np.array(embeddings)
        print(f"✅ Generated embeddings shape: {embeddings_array.shape}")
        
        return EmbeddingResult(
            embeddings=embeddings_array,
            chunk_ids=chunk_ids,
            metadata=metadata
        )
    
    def embed_query(self, query: str) -> np.ndarray:
        """Convert user query to embedding"""
        if self.model is None:
            self.load_model()
        
        # Add e5 prefix for queries
        query_text = f"query: {query}"
        embedding = self.model.encode(
            [query_text],
            convert_to_tensor=False,
            normalize_embeddings=True
        )
        return embedding[0]
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Monitor memory usage"""
        memory = psutil.virtual_memory()
        return {
            'total_gb': memory.total / (1024**3),
            'used_gb': memory.used / (1024**3),
            'available_gb': memory.available / (1024**3),
            'percent_used': memory.percent
        }