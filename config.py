"""
Configuration settings for the Biology RAG Chatbot
"""

import os
from pathlib import Path

# ===== API Configuration =====
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# Model settings
CHAT_MODEL = "gemini-2.5-flash"  # or "gemini-1.5-pro" for better quality
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"

# ===== Data Paths =====
DATA_DIR = Path("data")
PDF_PATH = "BrTr_ocr.pdf"  # Original PDF (for preprocessing only)

# ===== RAG System Settings =====
# Retrieval settings
TOP_K_RETRIEVE = 10  # Number of chunks to retrieve
TOP_K_PER_METHOD = 15  # Per retrieval method (dense/sparse)
MAX_CONTEXT_CHARS = 8000  # Maximum context length

# Chunking settings (for preprocessing)
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# ===== Streamlit UI Settings =====
APP_TITLE = "🎓 Biology Tutor - Bryophyta & Pteridophyta"
APP_SUBTITLE = "জীববিজ্ঞান শিক্ষক - ব্রায়োফাইটা ও টেরিডোফাইটা"
APP_ICON = "🌿"

# Chat settings
MAX_MESSAGES_DISPLAY = 50  # Maximum chat history to display
SHOW_SOURCES_DEFAULT = True  # Show sources by default
SHOW_THINKING_DEFAULT = False  # Show retrieval process

# UI Colors (Streamlit theme)
PRIMARY_COLOR = "#2E7D32"  # Green for biology theme
BACKGROUND_COLOR = "#F1F8E9"
SECONDARY_BG = "#C5E1A5"

# ===== System Messages =====
WELCOME_MESSAGE = """
স্বাগতম! আমি তোমার জীববিজ্ঞান শিক্ষক। ব্রায়োফাইটা ও টেরিডোফাইটা বিষয়ে যেকোনো প্রশ্ন করতে পারো।

**তুমি জিজ্ঞাসা করতে পারো:**
- রিকসিয়ার বৈশিষ্ট্য কী?
- What is the difference between bryophytes and pteridophytes?
- মস উদ্ভিদের জীবনচক্র ব্যাখ্যা করো

**ভাষা:** বাংলা, ইংরেজি, বা Banglish - যেকোনোটি!
"""

ERROR_NO_API_KEY = """
⚠️ **API Key Missing**

Please set your Gemini API key:
1. Create a `.streamlit/secrets.toml` file
2. Add: `GEMINI_API_KEY = "your-key-here"`

Or set environment variable: `GEMINI_API_KEY`
"""

ERROR_DATA_NOT_FOUND = """
⚠️ **Data Files Not Found**

Please run the preprocessing script first:
```bash
python preprocess.py
```

This will create the necessary index files in the `data/` folder.
"""

# ===== Performance Settings =====
# Caching
ENABLE_QUERY_CACHE = True
CACHE_TTL_SECONDS = 3600  # 1 hour

# Rate limiting (for production)
MAX_REQUESTS_PER_USER = 50  # Per session
MAX_REQUESTS_PER_MINUTE = 10

# ===== Development Settings =====
DEBUG_MODE = os.environ.get("DEBUG", "False").lower() == "true"
SHOW_RETRIEVAL_DEBUG = DEBUG_MODE
SHOW_PROMPT_DEBUG = DEBUG_MODE

# ===== Deployment Settings =====
# For Streamlit Cloud / Hugging Face Spaces
DEPLOYMENT_PLATFORM = os.environ.get("DEPLOYMENT_PLATFORM", "local")  # local, streamlit, huggingface

# Health check endpoint (for production monitoring)
ENABLE_HEALTH_CHECK = True
