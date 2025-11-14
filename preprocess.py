"""
Preprocessing Script: Build and save RAG indices
Run this ONCE before deploying the chatbot

This script:
1. Loads the PDF
2. Cleans OCR text
3. Creates semantic chunks
4. Generates embeddings
5. Builds FAISS and BM25 indices
6. Saves everything to disk for fast loading
"""

import os
import sys
import json
import pickle
import numpy as np
from pathlib import Path
import re

# Import required libraries
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
import faiss

from config import *


def clean_ocr_text(text):
    """Post-process OCR text to fix common errors"""
    if not text:
        return text
    
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    text = re.sub(r'[^\u0980-\u09FF\u0020-\u007E\s।,;:()"\'\-–—\n]', '', text)
    
    replacements = {
        'ও ০': '০',
        'া া': 'া',
        '  ': ' ',
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        valid_chars = len(re.findall(r'[\u0980-\u09FFa-zA-Z0-9]', line))
        total_chars = len(line.strip())
        
        if total_chars == 0 or (valid_chars / max(total_chars, 1)) > 0.5:
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines).strip()


def load_pdf_text(path):
    """Load and clean PDF text"""
    print(f"📄 Loading PDF: {path}")
    reader = PdfReader(path)
    pages = []
    
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = clean_ocr_text(text)
        
        if text and len(text.strip()) > 50:
            lines = text.split('\n')
            first_line = lines[0] if lines else ""
            
            pages.append({
                "page": i + 1,
                "text": text,
                "first_line": first_line[:100],
                "char_count": len(text),
                "line_count": len(lines)
            })
    
    print(f"   ✅ Loaded {len(pages)} pages")
    return pages


def semantic_chunking(text, page_num, chunk_size=800, overlap=150):
    """Create semantic chunks from text"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        separators=[
            "\n\n\n",
            "\n\n",
            "\n",
            "। ",
            "| ",
            " ",
            ""
        ],
        is_separator_regex=False,
    )
    
    chunks = splitter.split_text(text)
    
    enriched_chunks = []
    for i, chunk in enumerate(chunks):
        chunk = chunk.strip()
        if not chunk or len(chunk) < 50:
            continue
            
        has_list = bool(re.search(r'[\(১২৩৪৫৬৭৮৯০\d).][\s]*[\u0980-\u09FF]', chunk))
        has_heading = bool(re.search(r'^[A-Z\u0980-\u09FF]{3,}', chunk, re.MULTILINE))
        
        enriched_chunks.append({
            "page": page_num,
            "chunk_id": i,
            "text": chunk,
            "char_count": len(chunk),
            "has_list": has_list,
            "has_heading": has_heading,
        })
    
    return enriched_chunks


def tokenize_bangla(text):
    """Tokenize Bangla text for BM25"""
    tokens = re.findall(r'[\u0980-\u09FF]+|[a-zA-Z]+|\d+', text.lower())
    return tokens


def main():
    """Main preprocessing pipeline"""
    print("="*70)
    print("🔧 PREPROCESSING: Building RAG Indices")
    print("="*70)
    
    # Create data directory
    DATA_DIR.mkdir(exist_ok=True)
    
    # Check if PDF exists
    if not Path(PDF_PATH).exists():
        print(f"❌ ERROR: PDF not found at {PDF_PATH}")
        print("   Please update PDF_PATH in config.py or place PDF in current directory")
        sys.exit(1)
    
    # Step 1: Load PDF
    print("\n📖 Step 1: Loading PDF...")
    pages = load_pdf_text(PDF_PATH)
    
    # Step 2: Create chunks
    print("\n✂️  Step 2: Creating semantic chunks...")
    corpus_chunks = []
    for page in pages:
        chunks = semantic_chunking(page["text"], page["page"], CHUNK_SIZE, CHUNK_OVERLAP)
        corpus_chunks.extend(chunks)
    
    print(f"   ✅ Created {len(corpus_chunks)} chunks")
    print(f"   📊 Average chunk size: {sum(c['char_count'] for c in corpus_chunks) / len(corpus_chunks):.0f} chars")
    
    # Step 3: Generate embeddings
    print("\n🔢 Step 3: Generating embeddings...")
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    chunk_texts = [c["text"] for c in corpus_chunks]
    
    embeddings = embedder.encode(
        chunk_texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        batch_size=32
    )
    
    # Normalize embeddings
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings.astype("float32")
    
    print(f"   ✅ Generated embeddings: {embeddings.shape}")
    
    # Step 4: Build FAISS index
    print("\n🔍 Step 4: Building FAISS index...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"   ✅ FAISS index built with {index.ntotal} vectors")
    
    # Step 5: Build BM25 index
    print("\n📚 Step 5: Building BM25 index...")
    tokenized_corpus = [tokenize_bangla(c["text"]) for c in corpus_chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    print(f"   ✅ BM25 index built with {len(tokenized_corpus)} documents")
    
    # Step 6: Save everything
    print("\n💾 Step 6: Saving indices to disk...")
    
    # Save chunks
    with open(DATA_DIR / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(corpus_chunks, f, ensure_ascii=False, indent=2)
    print(f"   ✅ Saved chunks.json")
    
    # Save embeddings
    np.save(DATA_DIR / "embeddings.npy", embeddings)
    print(f"   ✅ Saved embeddings.npy")
    
    # Save FAISS index
    faiss.write_index(index, str(DATA_DIR / "faiss_index.bin"))
    print(f"   ✅ Saved faiss_index.bin")
    
    # Save BM25 data
    bm25_data = {
        "bm25": bm25,
        "tokenized_corpus": tokenized_corpus
    }
    with open(DATA_DIR / "bm25_data.pkl", "wb") as f:
        pickle.dump(bm25_data, f)
    print(f"   ✅ Saved bm25_data.pkl")
    
    # Summary
    print("\n" + "="*70)
    print("✨ PREPROCESSING COMPLETE!")
    print("="*70)
    print(f"📁 Data saved in: {DATA_DIR.absolute()}")
    print(f"📊 Total file size: {sum(f.stat().st_size for f in DATA_DIR.glob('*')) / 1024 / 1024:.2f} MB")
    print("\n🚀 You can now run the chatbot:")
    print("   streamlit run app.py")
    print("="*70)


if __name__ == "__main__":
    main()
