import sys
import os
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from document_processor import DocumentProcessor
from embedding_engine import EmbeddingEngine
from vector_db_manager import VectorDBManager

def setup_local_database():
    """Initialize local Qdrant database with ALL documents in the folder"""
    load_dotenv()
    
    print("🚀 Setting up local Qdrant database...")
    
    doc_processor = DocumentProcessor()
    embedding_engine = EmbeddingEngine()
    db_manager = VectorDBManager(
        url="http://localhost:6333",
        api_key=None
    )
    
    if not db_manager.test_connection():
        print("❌ Cannot connect to Qdrant Docker. Make sure container is running:")
        print("   docker-compose up -d")
        return False
    
    db_manager.setup_collection(force_recreate=True)
    
    documents_folder = 'documents'
    supported_extensions = ['.pdf', '.docx', '.doc']
    
    if not os.path.exists(documents_folder):
        print(f"❌ Documents folder not found: {documents_folder}")
        return False
    
    document_files = []
    for filename in os.listdir(documents_folder):
        file_path = os.path.join(documents_folder, filename)
        if os.path.isfile(file_path) and any(filename.lower().endswith(ext) for ext in supported_extensions):
            document_files.append(file_path)
    
    if not document_files:
        print(f"❌ No supported documents found in {documents_folder}")
        print(f"Supported formats: {', '.join(supported_extensions)}")
        return False
    
    print(f"📁 Found {len(document_files)} document(s) to process:")
    for doc_path in document_files:
        print(f"   - {doc_path}")
    
    all_chunks = []
    all_embeddings = []
    all_chunk_ids = []
    all_metadata = []
    
    for doc_path in document_files:
        try:
            print(f"\n📄 Processing document: {doc_path}")
            
            chunks = doc_processor.process_document(doc_path)
            print(f"✅ Generated {len(chunks)} chunks from {os.path.basename(doc_path)}")
            
            if chunks:
                embeddings_result = embedding_engine.embed_chunks(chunks)
                print(f"✅ Generated embeddings: {embeddings_result.embeddings.shape}")
                
                for chunk in chunks:
                    metadata = {
                        "content": chunk.content,
                        "filename": chunk.metadata.get("filename", os.path.basename(doc_path)),
                        "page_number": chunk.page_number,
                        "section_name": chunk.section_name,
                        "chunk_id": chunk.chunk_id
                    }
                    all_metadata.append(metadata)
                
                all_chunks.extend(chunks)
                all_embeddings.append(embeddings_result.embeddings)
                all_chunk_ids.extend(embeddings_result.chunk_ids)
            
        except Exception as e:
            print(f"❌ Error processing {doc_path}: {str(e)}")
            continue
    
    if all_embeddings:
        import numpy as np
        combined_embeddings = np.vstack(all_embeddings)
        
        print(f"\n📊 Total processing summary:")
        print(f"   - Documents processed: {len(document_files)}")
        print(f"   - Total chunks: {len(all_chunks)}")
        print(f"   - Combined embeddings shape: {combined_embeddings.shape}")
        
        print(f"\n💾 Storing {len(combined_embeddings)} embeddings in Qdrant...")
        db_manager.store_embeddings(
            combined_embeddings,
            all_chunk_ids,
            all_metadata
        )
        
        print("🎉 Local database setup completed successfully!")
        print("📊 Your RAG chatbot can now answer questions from ALL uploaded documents!")
        return True
    
    else:
        print("❌ No embeddings generated - no documents were processed successfully")
        return False

if __name__ == "__main__":
    setup_local_database()
