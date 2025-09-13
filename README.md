# 📞 AI-Powered Real Estate Call Bot

This project is an **AI voice assistant for real estate** that simulates property consultation calls.  
It uses **Generative AI (Gemini)** + **RAG (Retrieval Augmented Generation)** for smart responses, and can optionally integrate with **Twilio** for real phone conversations.

---

## 🚀 Features
- 🎙️ **Voice Assistant (chat.py)** → Local testing with speech recognition + TTS  
- 📊 **RAG Knowledge Base (properties_RAG.py)** → Context from property market data & company profile  
- ☁️ **Twilio Integration (twilio_webhookx.py)** → Real phone call support via Twilio Voice API  
- 🤖 **Generative AI (Gemini)** → Natural, persuasive, and context-aware conversation flow  
- 🔒 **Safe & Configurable** → API keys managed via `.env` file

---

## 📂 Project Structure

├── chat.py # Local AI voice assistant

├── properties_RAG.py # RAG knowledge base for real estate context

├── twilio_webhookx.py # Twilio webhook for real phone call integration

├── data/ # CSV & TXT knowledge base files

├── .env.example # Example environment variables

├── requirements.txt # Python dependencies

└── README.md # Project documentation



---

## ⚙️ Installation

1. **Clone the repo**
   ```bash
   git clone https://github.com/Keerthivasan96/AI-REAL-ESTATE-CALL-BOT.git
   cd AI-REAL-ESTATE-CALL-BOT


2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate   # (Linux/Mac)
   venv\Scripts\activate      # (Windows)
   
3.**Install dependencies**
   ```bash
   pip install -r requirements.txt






  
