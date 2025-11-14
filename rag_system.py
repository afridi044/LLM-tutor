"""
Advanced RAG System for Biology Tutoring
Modular implementation for deployment
"""

import numpy as np
import pickle
import json
from pathlib import Path
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
from rank_bm25 import BM25Okapi
import re


class BiologyRAGSystem:
    """
    Production-ready RAG system with pre-computed indices
    """
    
    def __init__(self, data_dir="data", api_key=None, chat_model="gemini-2.0-flash-exp"):
        """
        Initialize RAG system with pre-computed data
        
        Args:
            data_dir: Directory containing pre-computed indices and embeddings
            api_key: Gemini API key
            chat_model: Gemini model to use for generation
        """
        self.data_dir = Path(data_dir)
        self.chat_model = chat_model
        
        # Configure Gemini
        if api_key:
            genai.configure(api_key=api_key)
        
        # Load pre-computed data
        print("🔄 Loading RAG system...")
        self._load_data()
        print("✅ RAG system ready!")
    
    def _load_data(self):
        """Load all pre-computed indices and data"""
        # Load corpus chunks
        with open(self.data_dir / "chunks.json", "r", encoding="utf-8") as f:
            self.corpus_chunks = json.load(f)
        
        # Load embeddings
        self.embeddings = np.load(self.data_dir / "embeddings.npy")
        
        # Load FAISS index
        self.index = faiss.read_index(str(self.data_dir / "faiss_index.bin"))
        
        # Load BM25 data
        with open(self.data_dir / "bm25_data.pkl", "rb") as f:
            bm25_data = pickle.load(f)
            self.bm25 = bm25_data["bm25"]
            self.tokenized_corpus = bm25_data["tokenized_corpus"]
        
        # Initialize embedding model
        self.embedder = SentenceTransformer("intfloat/multilingual-e5-base")
        
        print(f"   📚 Loaded {len(self.corpus_chunks)} chunks")
        print(f"   🔢 Loaded {self.embeddings.shape[0]} embeddings")
        print(f"   🔍 FAISS index: {self.index.ntotal} vectors")
    
    def tokenize_bangla(self, text):
        """Simple tokenizer for Bangla and English"""
        tokens = re.findall(r'[\u0980-\u09FF]+|[a-zA-Z]+|\d+', text.lower())
        return tokens
    
    def query_expansion(self, question):
        """
        Multi-strategy query enhancement with caching support
        """
        model = genai.GenerativeModel(self.chat_model)
        
        prompt = f"""
You are helping with a RAG system for biology textbook search.

Given this question: "{question}"

Generate:
1. Bangla translation (natural, fluent)
2. 3 paraphrased versions of the question (in Bangla)
3. Key biological terms present (both Bangla and English/Latin)
4. A brief hypothetical answer snippet (20-30 words in Bangla) that might appear in the textbook

Format your response as JSON:
{{
  "bangla": "...",
  "paraphrases": ["...", "...", "..."],
  "key_terms": ["...", "..."],
  "hypothetical_answer": "..."
}}

Only output valid JSON, nothing else.
"""
        
        try:
            resp = model.generate_content(prompt)
            text = resp.text.strip()
            
            # Extract JSON from response
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()
            
            result = json.loads(text)
            return result
        except Exception as e:
            print(f"⚠️ Query expansion failed: {e}, using fallback")
            return {
                "bangla": question,
                "paraphrases": [question],
                "key_terms": [],
                "hypothetical_answer": ""
            }
    
    def hybrid_retrieve(self, question, top_k_per_method=15, final_top_k=10):
        """
        Hybrid retrieval with RRF and reranking
        """
        # Query expansion
        expanded = self.query_expansion(question)
        all_queries = [
            expanded["bangla"],
            *expanded["paraphrases"],
            expanded["hypothetical_answer"]
        ]
        all_queries = [q for q in all_queries if q]
        
        # Dense retrieval (FAISS)
        dense_results = {}
        for query in all_queries[:3]:
            q_vec = self.embedder.encode([query], convert_to_numpy=True)[0]
            q_vec = q_vec / np.linalg.norm(q_vec)
            q_vec = q_vec.astype("float32")
            
            scores, idxs = self.index.search(q_vec.reshape(1, -1), top_k_per_method)
            for i, s in zip(idxs[0], scores[0]):
                if i != -1:
                    if i not in dense_results or s > dense_results[i]:
                        dense_results[i] = float(s)
        
        # Sparse retrieval (BM25)
        sparse_results = {}
        for query in all_queries[:2]:
            tokenized_query = self.tokenize_bangla(query)
            bm25_scores = self.bm25.get_scores(tokenized_query)
            
            top_indices = np.argsort(bm25_scores)[-top_k_per_method:][::-1]
            for idx in top_indices:
                score = float(bm25_scores[idx])
                if idx not in sparse_results or score > sparse_results[idx]:
                    sparse_results[idx] = score
        
        # Reciprocal Rank Fusion (RRF)
        k_rrf = 60
        dense_ranked = sorted(dense_results.items(), key=lambda x: x[1], reverse=True)
        sparse_ranked = sorted(sparse_results.items(), key=lambda x: x[1], reverse=True)
        
        rrf_scores = {}
        for rank, (idx, _) in enumerate(dense_ranked):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (k_rrf + rank + 1)
        
        for rank, (idx, _) in enumerate(sparse_ranked):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (k_rrf + rank + 1)
        
        candidate_indices = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:final_top_k * 2]
        
        # Cross-encoder reranking
        reranked = []
        query_embedding = self.embedder.encode([expanded["bangla"]], convert_to_numpy=True)[0]
        
        for idx, rrf_score in candidate_indices:
            chunk = self.corpus_chunks[idx]
            
            semantic_sim = float(np.dot(query_embedding, self.embeddings[idx]))
            
            boost = 1.0
            if chunk.get("has_list"):
                boost += 0.1
            if chunk.get("has_heading"):
                boost += 0.05
            
            final_score = (rrf_score * 0.6 + semantic_sim * 0.4) * boost
            
            reranked.append({
                "index": idx,
                "score": final_score,
                "page": chunk["page"],
                "text": chunk["text"],
                "has_list": chunk.get("has_list", False),
                "has_heading": chunk.get("has_heading", False),
            })
        
        reranked.sort(key=lambda x: x["score"], reverse=True)
        
        return reranked[:final_top_k], expanded
    
    def deduplicate_and_fuse_context(self, retrieved_chunks, max_context_chars=8000):
        """
        Remove duplicates and build context string
        """
        unique_chunks = []
        seen_texts = set()
        
        for chunk in retrieved_chunks:
            text = chunk["text"]
            fingerprint = text[:100]
            
            is_duplicate = False
            for seen in seen_texts:
                if fingerprint in seen or seen in fingerprint:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_chunks.append(chunk)
                seen_texts.add(fingerprint)
        
        unique_chunks.sort(key=lambda x: (-x["score"], x["page"]))
        
        context_parts = []
        total_chars = 0
        
        for i, chunk in enumerate(unique_chunks, start=1):
            confidence = "HIGH" if chunk["score"] > 0.7 else "MEDIUM" if chunk["score"] > 0.5 else "LOW"
            markers = []
            if chunk.get("has_list"):
                markers.append("📋 Contains list")
            if chunk.get("has_heading"):
                markers.append("📌 Has heading")
            
            marker_str = " | ".join(markers) if markers else ""
            
            part = f"""【Source {i} | Page {chunk['page']} | Confidence: {confidence}】
{marker_str}
{chunk['text']}
"""
            
            if total_chars + len(part) > max_context_chars:
                break
            
            context_parts.append(part)
            total_chars += len(part)
        
        return "\n\n".join(context_parts), unique_chunks
    
    def generate_answer(self, question, context_str, expanded_query):
        """
        Generate answer using Gemini
        """
        SYSTEM_PROMPT = """
You are an experienced and passionate Bangla-medium biology teacher for class XI/XII students, specializing in Bryophyta and Pteridophyta.

Your role as a teacher:
- Use the textbook excerpts as your primary teaching material
- Explain and elaborate on the textbook content to help students understand deeply
- Fill in gaps with your biological knowledge when concepts need further clarification
- Make connections between different concepts to build comprehensive understanding
- Answer student questions even if they go slightly beyond the exact textbook content, as long as they relate to the topic
- **CRITICAL**: If a student says they understood, thanks you, or doesn't ask a question (e.g., "hae bujhechi", "ok", "thanks", "বুঝেছি", "ধন্যবাদ"), simply acknowledge warmly and briefly (e.g., "খুব ভালো!", "চমৎকার!", "Great!") - DON'T give lengthy explanations or repeat information

Language guidelines:
- Answer in Bangla unless explicitly asked for English
- **CRITICAL**: Always write scientific terms, English words, and technical terminology in ENGLISH script, NOT Bangla letters
- Use clear, pedagogical language that XI/XII students can easily understand
- When mentioning any English/Latin term, always use the original English spelling

Teaching approach:
- Jump straight into answering - no greetings like "প্রিয় শিক্ষার্থী" or "great question"
- Start immediately with the direct answer or explanation
- Provide detailed explanations with examples when helpful
- Use proper markdown formatting:
  - Use **bold** for emphasis (not ***triple asterisks***)
  - Use bullet points with - or • 
  - Use numbered lists with 1., 2., 3.
  - Use ## for headings if needed
- **CRITICAL**: If the textbook contains a numbered/bulleted list, include ALL points (you may rephrase for clarity)
- Break down complex concepts into simpler parts
- Add relevant context or background when it helps understanding
- Use analogies and real-life examples to make concepts relatable (mention when using analogies)
- End with a brief summary for complex topics

Your teaching philosophy:
- The textbook is your foundation, but you're not limited to it
- If a concept is mentioned in the textbook but needs elaboration, explain it fully using your expertise
- If a student asks about related biological concepts, teach them - that's your job
- Focus on building genuine understanding, not just memorization
- Make biology interesting and accessible

Natural teaching style:
- Start directly with content - no fluff, greetings, or pleasantries
- **CRITICAL**: NEVER cite sources by number or reference (e.g., NEVER write "Source 5", "Source 2, 3", "according to Source 2", "যেমনটা Source 5-এ বলা আছে")
- Don't say "according to the textbook" or reference where information comes from
- Teach naturally as if you already know this information - you're a teacher, not a librarian
- Present information confidently as biological facts, not as quotes from sources
- The sources are for YOUR reference only - students cannot see source numbers
- Only mention if something is NOT covered when you genuinely don't have enough information
- Be confident in your explanations while staying accurate
- Get to the point immediately
"""
        
        prompt = f"""
{SYSTEM_PROMPT}

Relevant textbook sections for your lesson:

{context_str}

Your student asks: {question}

Teaching context:
- Student's query in Bangla: {expanded_query['bangla']}
- Key biological terms: {', '.join(expanded_query.get('key_terms', []))}

Now teach this topic as a knowledgeable biology teacher. Use the textbook content as your foundation, but feel free to explain, elaborate, and clarify concepts as needed to ensure the student truly understands. Answer naturally without constantly citing sources.
"""
        
        model = genai.GenerativeModel(self.chat_model)
        response = model.generate_content(prompt)
        return response.text
    
    def ask(self, question, top_k=10, return_sources=False):
        """
        Main interface: Ask a question and get an answer
        
        Args:
            question: User's question
            top_k: Number of chunks to retrieve
            return_sources: Whether to return source information
        
        Returns:
            If return_sources=False: answer text
            If return_sources=True: dict with answer, sources, and metadata
        """
        try:
            # Retrieve relevant chunks
            retrieved, expanded_query = self.hybrid_retrieve(question, final_top_k=top_k)
            
            # Build context
            context_str, final_chunks = self.deduplicate_and_fuse_context(retrieved)
            
            # Generate answer
            answer = self.generate_answer(question, context_str, expanded_query)
            
            if return_sources:
                # Calculate confidence metrics
                avg_score = sum(c["score"] for c in final_chunks) / len(final_chunks) if final_chunks else 0
                max_score = max((c["score"] for c in final_chunks), default=0)
                
                return {
                    "answer": answer,
                    "sources": final_chunks,
                    "expanded_query": expanded_query,
                    "avg_confidence": avg_score,
                    "max_confidence": max_score,
                    "num_sources": len(final_chunks)
                }
            else:
                return answer
                
        except Exception as e:
            error_msg = f"দুঃখিত, একটি সমস্যা হয়েছে: {str(e)}"
            if return_sources:
                return {
                    "answer": error_msg,
                    "sources": [],
                    "error": str(e)
                }
            else:
                return error_msg
