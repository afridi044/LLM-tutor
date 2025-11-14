# 🎓 Biology Tutor Chatbot - Bryophyta & Pteridophyta

An intelligent chatbot powered by advanced RAG (Retrieval-Augmented Generation) for answering questions from a Bangla biology textbook.

## ✨ Features

- 🔍 **Hybrid Retrieval**: Combines dense (FAISS) and sparse (BM25) search with Reciprocal Rank Fusion
- 🌐 **Multi-language**: Supports Bangla, English, and Banglish queries
- 📚 **Source Citations**: Shows relevant textbook page references
- 💡 **Educational**: Provides detailed explanations with examples
- ⚡ **Fast**: Pre-computed indices for instant responses
- 🎨 **Beautiful UI**: Clean, intuitive Streamlit interface

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up API Key

Create a `.streamlit/secrets.toml` file:

```toml
GEMINI_API_KEY = "your-gemini-api-key-here"
```

Or set environment variable:

```bash
# Windows PowerShell
$env:GEMINI_API_KEY="your-key-here"

# Linux/Mac
export GEMINI_API_KEY="your-key-here"
```

Get your API key from: https://makersuite.google.com/app/apikey

### 3. Preprocess Data (One-time)

Place your PDF (`BrTr_ocr.pdf`) in the project folder, then run:

```bash
python preprocess.py
```

This will:
- Extract and clean text from PDF
- Create semantic chunks
- Generate embeddings
- Build search indices
- Save everything to `data/` folder

**Expected output:**
```
✨ PREPROCESSING COMPLETE!
📁 Data saved in: C:\path\to\data
📊 Total file size: XX.XX MB
```

### 4. Run the Chatbot

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

## 📁 Project Structure

```
LLM tutor/
├── app.py                 # Streamlit chat interface
├── rag_system.py          # Core RAG implementation
├── config.py              # Configuration settings
├── preprocess.py          # Data preprocessing script
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── BrTr_ocr.pdf          # Your textbook PDF
└── data/                  # Generated indices (after preprocessing)
    ├── chunks.json        # Text chunks with metadata
    ├── embeddings.npy     # Dense embeddings
    ├── faiss_index.bin    # FAISS vector index
    └── bm25_data.pkl      # BM25 sparse index
```

## 🎯 Usage Examples

**Questions you can ask:**

- `রিকসিয়ার বৈশিষ্ট্য কী?`
- `What is the difference between bryophytes and pteridophytes?`
- `মস উদ্ভিদের জীবনচক্র ব্যাখ্যা করো`
- `riccia er shonaktokari boishisto gulo bolo` (Banglish)

## ⚙️ Configuration

Edit `config.py` to customize:

- **Models**: Change `CHAT_MODEL` or `EMBEDDING_MODEL`
- **Retrieval**: Adjust `TOP_K_RETRIEVE`, `CHUNK_SIZE`, etc.
- **UI**: Modify `APP_TITLE`, colors, welcome message
- **Performance**: Enable caching, rate limiting

## 🌐 Deployment

### Option 1: Streamlit Community Cloud (Free)

1. Push code to GitHub
2. Go to https://share.streamlit.io
3. Deploy from your repository
4. Add `GEMINI_API_KEY` in Secrets management

**Important**: Upload the `data/` folder to your repo (or use GitHub LFS for large files)

### Option 2: Hugging Face Spaces (Free)

1. Create a new Space at https://huggingface.co/spaces
2. Choose "Streamlit" as SDK
3. Upload all files including `data/` folder
4. Add `GEMINI_API_KEY` in Settings → Secrets

### Option 3: Local Network

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Access from other devices: `http://your-ip:8501`

### Option 4: Docker (Production)

Create `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"]
```

Build and run:

```bash
docker build -t biology-tutor .
docker run -p 8501:8501 -e GEMINI_API_KEY=your-key biology-tutor
```

## 🔧 Troubleshooting

### "Data files not found"
- Run `python preprocess.py` first
- Ensure `data/` folder exists with all 4 files

### "API key missing"
- Set `GEMINI_API_KEY` in `.streamlit/secrets.toml` or environment
- Check key is valid at https://makersuite.google.com

### Slow performance
- Reduce `TOP_K_RETRIEVE` in sidebar (try 5 instead of 10)
- Use `gemini-2.0-flash-exp` instead of `gemini-1.5-pro`
- Ensure preprocessing completed successfully

### Out of memory
- Reduce `CHUNK_SIZE` in `config.py`
- Use smaller embedding model (e.g., `paraphrase-multilingual-MiniLM-L12-v2`)

## 📊 System Requirements

- **RAM**: 2GB minimum (4GB recommended)
- **Disk**: ~500MB for models + data
- **Python**: 3.9+
- **Internet**: Required for Gemini API calls

## 🔒 Security Notes

- Never commit `.streamlit/secrets.toml` to GitHub
- Add to `.gitignore`:
  ```
  .streamlit/secrets.toml
  .env
  ```
- Use environment variables in production
- Consider rate limiting for public deployments

## 📈 Performance Tips

1. **Pre-compute everything**: Run `preprocess.py` offline
2. **Use caching**: Streamlit caches the RAG system automatically
3. **Optimize queries**: Shorter, focused questions work best
4. **Batch processing**: Use `data/` folder with pre-built indices
5. **Model selection**: `gemini-2.0-flash-exp` is 10x faster than `gemini-1.5-pro`

## 🤝 Contributing

Feel free to:
- Add more textbooks (modify `preprocess.py`)
- Improve chunking strategies (`rag_system.py`)
- Enhance UI (`app.py`)
- Add new features (export chat, PDF viewer, etc.)

## 📝 License

This project is for educational purposes. Ensure you have rights to the textbook content.

## 🆘 Support

Issues? Questions?
- Check `config.py` for settings
- Review logs in terminal
- Ensure all dependencies installed
- Verify API key is valid

## 🎓 How It Works

1. **Preprocessing** (one-time):
   - PDF → Text extraction → OCR cleaning
   - Text → Semantic chunks (800 chars)
   - Chunks → Dense embeddings (768-dim vectors)
   - Build FAISS index (fast similarity search)
   - Build BM25 index (keyword search)

2. **Query time** (real-time):
   - User question → Query expansion (translation, paraphrasing, HyDE)
   - Parallel retrieval: Dense (FAISS) + Sparse (BM25)
   - Reciprocal Rank Fusion → combine results
   - Cross-encoder reranking → final top-K
   - Context assembly → LLM generation (Gemini)
   - Answer + source citations

## 🔬 RAG Architecture

```
┌─────────────┐
│ User Query  │
└──────┬──────┘
       │
       ▼
┌──────────────────┐
│ Query Expansion  │ (Gemini)
│ - Translation    │
│ - Paraphrasing   │
│ - HyDE           │
└────────┬─────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌────────┐
│ Dense  │ │ Sparse │
│ FAISS  │ │  BM25  │
└────┬───┘ └───┬────┘
     │         │
     └────┬────┘
          ▼
    ┌──────────┐
    │   RRF    │
    │ Fusion   │
    └─────┬────┘
          │
          ▼
    ┌──────────┐
    │ Reranking│
    └─────┬────┘
          │
          ▼
    ┌──────────┐
    │ Context  │
    │ Assembly │
    └─────┬────┘
          │
          ▼
    ┌──────────┐
    │  Gemini  │
    │   LLM    │
    └─────┬────┘
          │
          ▼
    ┌──────────┐
    │  Answer  │
    │ +Sources │
    └──────────┘
```

---

**Built with ❤️ using Streamlit, Gemini, and state-of-the-art RAG techniques**
