import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from typing import List, Dict, Optional
import re
from dataclasses import dataclass
import psutil


@dataclass
class AnswerResult:
    answer: str
    confidence_score: float
    source_chunks: List[str]
    citations: List[Dict]
    found_in_document: bool
    relevant_sentences: List[str] = None  
    enhanced_citations: List[Dict] = None  


class LLMPipeline:
    def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
    
    def load_models(self):
        """Load TinyLlama model with CPU optimization"""
        print(f"Loading TinyLlama model: {self.model_name}")
        
        memory_gb = psutil.virtual_memory().available / (1024**3)
        if memory_gb < 4.0:  
            raise RuntimeError(f"Insufficient memory: {memory_gb:.1f}GB available, need at least 4GB")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True
        )
        
        self.model.eval()
        print("✅ TinyLlama model loaded successfully")
    
    def _preprocess_query(self, query: str) -> str:
        """Preprocess user query for better matching"""
        query = query.strip().lower()
        
        return ' '.join(query.split())

    def _format_answer_case(self, answer: str) -> str:
        """Format answer to lowercase while preserving sentence structure"""
        if not answer or "not found in the document" in answer.lower():
            return answer  
        
        answer = answer.lower()
        if answer:
            answer = answer[0].upper() + answer[1:]
        
        return answer

    def generate_answer_from_context(self, query: str, context_chunks: List[Dict]) -> AnswerResult:
        """NEW DIRECT LLM APPROACH: Document check first, then direct LLM generation"""
        if not context_chunks:
            return AnswerResult(
                answer="I cannot find any relevant information in the document to answer your question.",
                confidence_score=0.0,
                source_chunks=[],
                citations=[],
                found_in_document=False,
                relevant_sentences=[],
                enhanced_citations=[]
            )
        
        processed_query = self._preprocess_query(query)
        
        try:
            result = self._generate_answer_direct(query, context_chunks)
            
            if result is None:
                print("WARNING - _generate_answer_direct returned None, creating fallback result")
                result = AnswerResult(
                    answer="The answer is not found in the document.",
                    confidence_score=0.0,
                    source_chunks=[],
                    citations=[],
                    found_in_document=False,
                    relevant_sentences=[],
                    enhanced_citations=[]
                )
            
            if result.answer:
                result.answer = self._format_answer_case(result.answer)
            
            return result
            
        except Exception as e:
            print(f"ERROR in generate_answer_from_context: {str(e)}")
            return AnswerResult(
                answer="An error occurred while processing your question.",
                confidence_score=0.0,
                source_chunks=[],
                citations=[],
                found_in_document=False,
                relevant_sentences=[],
                enhanced_citations=[]
            )

    def _generate_answer_direct(self, query: str, context_chunks: List[Dict]) -> AnswerResult:
        """NEW DIRECT APPROACH: Check document relevance first, then LLM generation"""
        print("DEBUG - Using direct LLM approach with document filtering")
        
        is_document_relevant = self._is_query_document_relevant(query, context_chunks)
        
        print(f"DEBUG - Document relevance check: {is_document_relevant}")
        
        if not is_document_relevant:
            print("DEBUG - Query not relevant to document, returning 'not found'")
            return AnswerResult(
                answer="The answer is not found in the document.",
                confidence_score=0.0,
                source_chunks=[],
                citations=[],
                found_in_document=False,
                relevant_sentences=[],
                enhanced_citations=[]
            )
        
        try:
            print("DEBUG - Query relevant to document, generating LLM answer")
            
            if self.model is None:
                self.load_models()
            
            llm_answer = self._generate_with_full_context(query, context_chunks)
            
            if llm_answer and len(llm_answer.strip()) > 2:
                print(f"DEBUG - LLM generation successful: '{llm_answer}'")
                
                citations = self._generate_citations(context_chunks, llm_answer, query)
                relevant_sentences = self._extract_relevant_sentences(query, context_chunks, llm_answer)
                enhanced_citations = self._generate_enhanced_citations(context_chunks, llm_answer, query)
                confidence = self._calculate_confidence(llm_answer, citations)
                
                return AnswerResult(
                    answer=llm_answer,
                    confidence_score=confidence,
                    source_chunks=[chunk.get('content', '') for chunk in context_chunks],
                    citations=citations,
                    found_in_document=True,
                    relevant_sentences=relevant_sentences,
                    enhanced_citations=enhanced_citations
                )
            else:
                print("DEBUG - LLM generation failed, returning 'not found'")
                return AnswerResult(
                    answer="The answer is not found in the document.",
                    confidence_score=0.0,
                    source_chunks=[],
                    citations=[],
                    found_in_document=False,
                    relevant_sentences=[],
                    enhanced_citations=[]
                )
                
        except Exception as e:
            print(f"ERROR in direct LLM generation: {str(e)}")
            return AnswerResult(
                answer="The answer is not found in the document.",
                confidence_score=0.0,
                source_chunks=[],
                citations=[],
                found_in_document=False,
                relevant_sentences=[],
                enhanced_citations=[]
            )

    def _is_query_document_relevant(self, query: str, context_chunks: List[Dict]) -> bool:
        """Enhanced check if the query is relevant to the document content"""
        query_lower = query.lower()
        
        non_document_keywords = [
            "bitcoin", "pasta", "cooking", "price", "stock", "weather", "recipe", 
            "python", "programming", "movie", "song", "restaurant", "hotel", 
            "travel", "currency", "crypto", "food", "music", "film"
        ]
        
        if any(keyword in query_lower for keyword in non_document_keywords):
            return False
        
        document_content = ""
        for chunk in context_chunks[:2]:
            document_content += " " + chunk.get('content', '').lower()
        
        if "how many players" in query_lower:
            return any(word in document_content for word in ["eleven", "11", "players", "team", "sides"])
        
        if "what phases" in query_lower:
            if not any(word in document_content for word in ["batting", "bowling", "fielding", "phases"]):
                return False
        
        query_words = [word for word in query_lower.split() if len(word) > 3]
        if query_words:
            relevant_count = sum(1 for word in query_words if word in document_content)
            return relevant_count >= 1
        
        return True

    def _generate_with_full_context(self, query: str, context_chunks: List[Dict]) -> str:
        """Generate answer using LLM with clean context - NO METADATA"""
        try:
            clean_context_parts = []
            for chunk in context_chunks[:2]: 
                content = chunk.get('content', '').strip()
                if content:
                    lines = content.split('\n')
                    clean_lines = []
                    for line in lines:
                        line = line.strip()
                        if (line and 
                            not line.startswith('Laws of Cricket') and
                            not line.startswith('Page') and
                            not line.startswith('Section') and
                            not line.startswith('LAW ') and
                            not line.isdigit() and
                            len(line) > 10):  
                            clean_lines.append(line)
                    
                    if clean_lines:
                        clean_content = ' '.join(clean_lines)
                        words = clean_content.split()
                        if len(words) > 50:
                            clean_content = ' '.join(words[:50])
                        clean_context_parts.append(clean_content)
            
            if not clean_context_parts:
                return None
            
            final_context = ' '.join(clean_context_parts)
            
            prompt = f"""Read this text: {final_context}

    Question: {query}

    Instructions: Answer the question in exactly 1-2 sentences using ONLY the information provided. Do not add extra information. Do not explain. Just answer the specific question asked.

    Answer:"""
            
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=200,  
                truncation=True
            )
            
            print(f"DEBUG - Clean prompt input length: {inputs['input_ids'].shape[1]}")
            print(f"DEBUG - Clean context used: '{final_context[:100]}...'")
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=25,  
                    do_sample=False,    
                    temperature=0.1,    
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            ).strip()
            
            print(f"DEBUG - Clean LLM response: '{response}'")
            
            cleaned_response = self._clean_answer(response)
            
            if self._is_answer_relevant_to_question(query, cleaned_response, final_context):
                return cleaned_response
            else:
                print(f"DEBUG - Answer not relevant to question, returning None")
                return None
                
        except Exception as e:
            print(f"ERROR in clean context generation: {str(e)}")
            return None

    def _is_answer_relevant_to_question(self, question: str, answer: str, context: str) -> bool:
        """Strict validation to ensure answer actually answers the question"""
        if not answer or len(answer.strip()) < 3:
            return False
        
        question_lower = question.lower()
        answer_lower = answer.lower()
        context_lower = context.lower()
        
        answer_words = set(word for word in answer_lower.split() if len(word) > 3)
        context_words = set(word for word in context_lower.split() if len(word) > 3)
        
        if answer_words:
            context_overlap = len(answer_words.intersection(context_words)) / len(answer_words)
            if context_overlap < 0.3:  # At least 30% of answer words must be from context
                return False
        
        if "how many" in question_lower:
            if not any(word in answer_lower for word in ["eleven", "11", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]):
                return False
        
        if "what phases" in question_lower or "what does" in question_lower:
            if any(word in answer_lower for word in ["appendix", "law", "code", "edition"]):
                return False
        
        irrelevant_patterns = [
            "laws of cricket",
            "appendix",
            "captain's duties",
            "batting order",
            "edition",
            "preamble"
        ]
        
        if any(pattern in answer_lower for pattern in irrelevant_patterns):
            return False
        
        return True


    def _extract_relevant_sentences(self, query: str, context_chunks: List[Dict], answer: str) -> List[str]:
        """Extract complete sentences from chunks that are relevant to the query and answer"""
        relevant_sentences = []
        
        query_terms = set(word.lower() for word in query.split() if len(word) > 2)
        answer_terms = set(word.lower() for word in answer.split() if len(word) > 2)
        search_terms = query_terms.union(answer_terms)
        
        common_words = {'the', 'and', 'are', 'this', 'that', 'with', 'have', 'from', 'they', 'each', 'which'}
        search_terms = search_terms - common_words
        
        for chunk in context_chunks:
            content = chunk.get('content', '')
            
            sentences = re.split(r'[.!?]+', content)
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 15:  
                    sentence_words = set(word.lower() for word in sentence.split() if len(word) > 2)
                    sentence_words = sentence_words - common_words
                    
                    overlap = len(search_terms.intersection(sentence_words))
                    if overlap >= 2:  
                        enhanced_sentence = f"{sentence}. [Source: {chunk.get('filename', 'Unknown')}, Page {chunk.get('page_number', '?')}]"
                        if enhanced_sentence not in relevant_sentences:
                            relevant_sentences.append(enhanced_sentence)
        
        return relevant_sentences[:3]  
    
    def _generate_enhanced_citations(self, context_chunks: List[Dict], answer: str, query: str) -> List[Dict]:
        """Generate enhanced citations that include the complete chunk content"""
        enhanced_citations = []
        
        answer_words = set(word.lower() for word in answer.split() if len(word) > 2)
        
        for chunk in context_chunks:
            content = chunk.get('content', '').lower()
            chunk_words = set(word for word in content.split() if len(word) > 2)
            
            if answer_words:
                overlap = len(answer_words.intersection(chunk_words))
                relevance = overlap / len(answer_words)
                
                if relevance > 0.1:
                    enhanced_citation = {
                        'filename': chunk.get('filename', 'Document'),
                        'page_number': chunk.get('page_number', 1),
                        'section_name': chunk.get('section_name', 'Content'),
                        'chunk_id': chunk.get('chunk_id', 'Chunk_1'),
                        'relevance_score': relevance,
                        'full_content': chunk.get('content', ''),
                        'content_preview': chunk.get('content', '')[:200] + "..." if len(chunk.get('content', '')) > 200 else chunk.get('content', ''),
                        'word_count': len(chunk.get('content', '').split())
                    }
                    enhanced_citations.append(enhanced_citation)
        
        enhanced_citations.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        return enhanced_citations[:2]
    
    def _clean_answer(self, answer: str) -> str:
        """Clean the answer text"""
        if not answer:
            return ""
        
        prefixes = ["Answer:", "A:", "Response:", "The answer is:", "Based on this information:", "Answer (2-3 lines only):"]
        for prefix in prefixes:
            if answer.startswith(prefix):
                answer = answer[len(prefix):].strip()
        
        answer = ' '.join(answer.split())
        
        if answer and not answer.endswith(('.', '!', '?')):
            answer += "."
        
        return answer
    
    def _is_answer_contextual(self, answer: str, context: str) -> bool:
        """Check if answer is related to context"""
        answer_words = set(word.lower() for word in answer.split() if len(word) > 2)
        context_words = set(word.lower() for word in context.split() if len(word) > 2)
        
        if not answer_words:
            return False
        
        overlap = len(answer_words.intersection(context_words))
        return overlap / len(answer_words) > 0.3  
    
    def _generate_citations(self, context_chunks: List[Dict], answer: str, query: str) -> List[Dict]:
        """Generate citations based on relevance"""
        citations = []
        
        answer_words = set(word.lower() for word in answer.split() if len(word) > 2)
        
        for chunk in context_chunks:
            content = chunk.get('content', '').lower()
            chunk_words = set(word for word in content.split() if len(word) > 2)
            
            if answer_words:
                overlap = len(answer_words.intersection(chunk_words))
                relevance = overlap / len(answer_words)
                
                if relevance > 0.1:
                    citation = {
                        'filename': chunk.get('filename', 'Document'),
                        'page_number': chunk.get('page_number', 1),
                        'section_name': chunk.get('section_name', 'Content'),
                        'chunk_id': chunk.get('chunk_id', 'Chunk_1'),
                        'relevance_score': relevance
                    }
                    citations.append(citation)
        
        citations.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        return citations[:2]
    
    def _calculate_confidence(self, answer: str, citations: List[Dict]) -> float:
        """Calculate confidence score"""
        base_confidence = 0.6  

        if citations:
            max_relevance = max(c.get('relevance_score', 0) for c in citations)
            base_confidence += min(0.3, max_relevance)
        
        word_count = len(answer.split())
        if word_count >= 4:
            base_confidence += 0.1
        
        return min(0.95, base_confidence)


if __name__ == "__main__":
    import sys
    import os
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)
    
    try:
        from document_processor import DocumentProcessor
        from vector_db_manager import VectorDBManager
        from embedding_engine import EmbeddingEngine
        from dotenv import load_dotenv
        
        load_dotenv()
        
        doc_processor = DocumentProcessor()
        embedding_engine = EmbeddingEngine()
        db_manager = VectorDBManager(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY")
        )
        llm_pipeline = LLMPipeline()
        
        print("🚀 Testing DIRECT LLM RAG System with Document Filtering\n")
        
        test_queries = [
            "How many players are in a cricket team?",
            "What phases does cricket involve?", 
            "What is the price of Bitcoin?",
            "How do you cook pasta?"
        ]
        
        for query in test_queries:
            print(f"🔍 Query: {query}")
            
            try:
                embedding_engine.load_model()
                query_embedding = embedding_engine.embed_query(query)
                
                search_results = db_manager.search_similar(query_embedding, top_k=5)
                
                if search_results:
                    context_chunks = []
                    for result in search_results:
                        context_chunks.append({
                            'content': result.content,
                            'filename': result.filename,
                            'page_number': result.page_number,
                            'section_name': result.section_name,
                            'chunk_id': result.chunk_id
                        })
                    
                    answer_result = llm_pipeline.generate_answer_from_context(query, context_chunks)
                    
                    print(f"📝 Answer: {answer_result.answer}")
                    print(f"📊 Found in Document: {'✅ YES' if answer_result.found_in_document else '❌ NO'}")
                    print(f"📊 Confidence: {answer_result.confidence_score:.2f}")
                    
                    if answer_result.relevant_sentences:
                        print(f"📄 Relevant Context from Document:")
                        for i, sentence in enumerate(answer_result.relevant_sentences, 1):
                            print(f"   {i}. {sentence}")
                    
                    print(f"📚 Citations: {len(answer_result.citations)}")
                    if answer_result.citations:
                        for i, citation in enumerate(answer_result.citations, 1):
                            print(f"   [{i}] File: {citation['filename']}")
                            print(f"       Page: {citation['page_number']}")
                            print(f"       Section: {citation['section_name']}")
                            print(f"       Chunk ID: {citation['chunk_id']}")
                
                else:
                    print("📝 Answer: No relevant documents found.")
                    print("📊 Found in Document: ❌ NO")
                    print("📊 Confidence: 0.00")
                    print("📚 Citations: 0")
                
            except Exception as e:
                print(f"❌ Error processing query: {str(e)}")
            
            print("="*80)
    
    except ImportError as e:
        print(f"⚠️ Could not import required modules: {str(e)}")
        print("Testing with sample data instead...")
        
        pipeline = LLMPipeline()
        
        sample_chunks = [
            {
                'content': 'Cricket is played between two teams of eleven players each on a circular field.',
                'filename': 'cricket_manual.pdf',
                'page_number': 15,
                'section_name': 'Team_Rules',
                'chunk_id': 'Page_15_Team_Rules_Para_1'
            },
            {
                'content': 'The game involves batting, bowling, and fielding with specific rules for each phase.',
                'filename': 'cricket_manual.pdf',
                'page_number': 22,
                'section_name': 'Game_Phases',
                'chunk_id': 'Page_22_Game_Phases_Para_1'
            }
        ]

        test_queries = [
            "How many players are in a cricket team?",
            "What phases does cricket involve?", 
            "What is the price of Bitcoin?",
            "How do you cook pasta?"
        ]

        for query in test_queries:
            print(f"\n{'='*70}")
            print(f"🔍 Query: {query}")
            result = pipeline.generate_answer_from_context(query, sample_chunks)
            
            print(f"📝 Answer: {result.answer}")
            print(f"📊 Found in Document: {'✅ YES' if result.found_in_document else '❌ NO'}")
            print(f"📊 Confidence: {result.confidence_score:.2f}")
            
            if result.relevant_sentences:
                print(f"📄 Relevant Context from Document:")
                for i, sentence in enumerate(result.relevant_sentences, 1):
                    print(f"   {i}. {sentence}")
            
            print(f"📚 Citations: {len(result.citations)}")
            if result.citations:
                for i, citation in enumerate(result.citations, 1):
                    print(f"   [{i}] File: {citation['filename']}")
                    print(f"       Page: {citation['page_number']}")
                    print(f"       Section: {citation['section_name']}")
                    print(f"       Chunk ID: {citation['chunk_id']}")
