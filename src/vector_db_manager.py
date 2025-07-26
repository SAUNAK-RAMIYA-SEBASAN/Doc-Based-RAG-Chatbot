import os
from typing import List, Dict, Tuple, Optional
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from qdrant_client.http.exceptions import ResponseHandlingException
from dataclasses import dataclass
import uuid
from tqdm import tqdm
import time


@dataclass
class SearchResult:
    chunk_id: str
    content: str
    score: float
    metadata: Dict
    page_number: int
    section_name: str
    filename: str


class VectorDBManager:
    def __init__(self, url: str = "http://localhost:6333", api_key: str = None, collection_name: str = "rag_documents"):
        if api_key:
            self.client = QdrantClient(url=url, api_key=api_key)
        else:
            self.client = QdrantClient(url=url)
            
        self.collection_name = collection_name
        self.embedding_dim = 384 
        
    def setup_collection(self, force_recreate: bool = False):
        """Create or recreate the Qdrant collection"""
        try:
            collections = self.client.get_collections().collections
            collection_exists = any(col.name == self.collection_name for col in collections)
            
            if collection_exists and force_recreate:
                print(f"Deleting existing collection: {self.collection_name}")
                self.client.delete_collection(self.collection_name)
                collection_exists = False
            
            if not collection_exists:
                print(f"Creating new collection: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dim,
                        distance=Distance.COSINE
                    )
                )
                print("✅ Collection created successfully")
            else:
                print(f"✅ Using existing collection: {self.collection_name}")
                
        except Exception as e:
            print(f"❌ Error setting up collection: {str(e)}")
            raise
    
    def store_embeddings(self, embeddings: np.ndarray, chunk_ids: List[str], metadata_list: List[Dict]):
        """Store embeddings with metadata in Qdrant"""
        try:
            print(f"Storing {len(embeddings)} embeddings in Qdrant...")
            
            points = []
            for i, (embedding, chunk_id, metadata) in enumerate(zip(embeddings, chunk_ids, metadata_list)):
                point = PointStruct(
                    id=str(uuid.uuid4()),  
                    vector=embedding.tolist(),
                    payload={
                        "chunk_id": chunk_id,
                        "content": metadata.get("content", ""),
                        "filename": metadata.get("filename", ""),
                        "page_number": metadata.get("page_number", 1),
                        "section_name": metadata.get("section_name", ""),
                        "metadata": metadata
                    }
                )
                points.append(point)
            
            batch_size = 50
            for i in tqdm(range(0, len(points), batch_size), desc="Uploading to Qdrant"):
                batch = points[i:i + batch_size]
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )
            
            print(f"✅ Successfully stored {len(points)} embeddings")
            
        except Exception as e:
            print(f"❌ Error storing embeddings: {str(e)}")
            raise
    
    def search_similar(self, query_embedding: np.ndarray, top_k: int = 5, 
                      filename_filter: str = None) -> List[SearchResult]:
        """Search for similar chunks based on query embedding"""
        try:
            query_filter = None
            if filename_filter:
                query_filter = Filter(
                    must=[
                        FieldCondition(
                            key="filename",
                            match=MatchValue(value=filename_filter)
                        )
                    ]
                )
            
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                query_filter=query_filter,
                limit=top_k,
                with_payload=True
            )
            
            results = []
            for result in search_results:
                search_result = SearchResult(
                    chunk_id=result.payload.get("chunk_id", ""),
                    content=result.payload.get("content", ""),
                    score=result.score,
                    metadata=result.payload.get("metadata", {}),
                    page_number=result.payload.get("page_number", 1),
                    section_name=result.payload.get("section_name", ""),
                    filename=result.payload.get("filename", "")
                )
                results.append(search_result)
            
            return results
            
        except Exception as e:
            print(f"❌ Error searching embeddings: {str(e)}")
            return []
    
    def get_collection_info(self) -> Dict:
        """Get information about the collection"""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": info.config.params.vectors.size,
                "vectors_count": info.points_count,
                "indexed": info.status,
                "distance": info.config.params.vectors.distance.value
            }
        except Exception as e:
            print(f"❌ Error getting collection info: {str(e)}")
            return {}
    
    def delete_collection(self):
        """Delete the entire collection"""
        try:
            self.client.delete_collection(self.collection_name)
            print(f"✅ Collection {self.collection_name} deleted successfully")
        except Exception as e:
            print(f"❌ Error deleting collection: {str(e)}")
    
    def test_connection(self) -> bool:
        """Test connection to Qdrant instance"""
        try:
            collections = self.client.get_collections()
            print("✅ Successfully connected to Qdrant instance")
            print(f"Available collections: {len(collections.collections)}")
            return True
        except Exception as e:
            print(f"❌ Failed to connect to Qdrant: {str(e)}")
            print("💡 Make sure Docker container is running: docker-compose up -d")
            return False

