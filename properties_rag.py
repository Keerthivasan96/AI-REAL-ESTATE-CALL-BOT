"""
properties_RAG.py - Retrieval Augmented Generation for Real Estate Assistant
----------------------------------------------------------------------------
This module builds and queries a knowledge base for real estate data
using LangChain, FAISS, and Gemini embeddings.

Features:
- Loads CSV and TXT files from the local `data/` folder
- Builds a FAISS vector database
- Provides context-aware answers for property-related queries
- Includes fallback for company profile questions

Note:
- Place your CSV/TXT knowledge files in the `data/` folder
- Configure your API key in `.env`
"""

import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import CSVLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# File paths (use local `data/` folder instead of absolute paths)
DATA_FILES = {
    "csv": [
        "data/dubai_property_index.csv",
        "data/summary_transactions.csv",
        "data/valuations1_rag.csv",
        "data/valuation2_rag.csv",
        "data/rents_rag.csv",
        "data/residential_sale_index_rag.csv",
        "data/faq_questions.csv",
    ],
    "txt": [
        "data/about_company.txt",
        "data/dubai_real_estate_market_analysis.txt",
        "data/about_the_bot.txt",
    ],
}

COMPANY_PROFILE_PATH = "data/about_company.txt"


class RealEstateRAG:
    """
    Smart context-aware RAG system for RealEstateCo.
    Uses market data, property data, and official company profile
    to answer natural questions.
    """

    def __init__(self):
        self.INDEX_DIR = "faiss_index"
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.vectorstore = self._initialize_vectorstore()
        self.company_profile = self._load_company_profile()

    def _initialize_vectorstore(self):
        """Load or build the FAISS vector store"""
        if os.path.exists(self.INDEX_DIR):
            print(f"✅ Loading existing FAISS index from {self.INDEX_DIR}")
            return FAISS.load_local(
                self.INDEX_DIR,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )

        print("📦 No index found. Building new FAISS index...")
        docs = self._load_all_documents()
        print(f"📄 Loaded total {len(docs)} raw documents before splitting.")

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)
        print(f"🔍 Created {len(chunks)} text chunks for embedding.")

        texts = [chunk.page_content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]

        print("🔧 Creating FAISS vectorstore...")
        vectorstore = FAISS.from_texts(texts, self.embeddings, metadatas=metadatas)
        vectorstore.save_local(self.INDEX_DIR)
        print(f"✅ Saved FAISS index to {self.INDEX_DIR}")
        return vectorstore

    def _load_all_documents(self):
        """Load all CSV + TXT docs"""
        docs = []
        docs.extend(self._load_csvs(DATA_FILES["csv"]))
        docs.extend(self._load_txts(DATA_FILES["txt"]))
        return docs

    def _load_csvs(self, filepaths):
        """Load CSV files from data/"""
        docs = []
        for path in filepaths:
            if os.path.exists(path):
                try:
                    loaded = CSVLoader(file_path=path).load()
                    print(f"📊 Loaded {len(loaded)} docs from: {os.path.basename(path)}")
                    docs.extend(loaded)
                except Exception as e:
                    print(f"❌ Failed to load CSV {path}: {e}")
            else:
                print(f"⚠️ CSV not found: {path}")
        return docs

    def _load_txts(self, filepaths):
        """Load TXT files from data/"""
        docs = []
        for path in filepaths:
            if os.path.exists(path):
                try:
                    loader = TextLoader(path, encoding="utf-8")
                    loaded = loader.load()
                    print(f"📄 Loaded {len(loaded)} docs from: {os.path.basename(path)}")
                    docs.extend(loaded)
                except Exception as e:
                    print(f"❌ Failed to load TXT {path}: {e}")
            else:
                print(f"⚠️ TXT not found: {path}")
        return docs

    def _load_company_profile(self):
        """Load the company profile (for brand-related queries)"""
        if os.path.exists(COMPANY_PROFILE_PATH):
            try:
                return TextLoader(COMPANY_PROFILE_PATH, encoding="utf-8").load()[0].page_content
            except Exception as e:
                print(f"❌ Could not load company profile: {e}")
                return ""
        else:
            print("⚠️ Company profile not found.")
            return ""

    def query_knowledge_base(self, query: str, k: int = 2) -> str:
        """
        Smart RAG query: 
        - Company-related questions → company profile
        - Else → vectorstore search + Gemini
        """
        query_lower = query.lower()
        if any(
            keyword in query_lower
            for keyword in ["company", "realestateco", "founder", "ceo", "head office"]
        ):
            return self._generate_company_response(query)

        results = self.vectorstore.similarity_search(query, k=k)
        context = "\n\n".join([doc.page_content for doc in results])

        if not context.strip():
            return "Sorry, I couldn't find relevant data. Would you like me to connect you with an advisor?"

        prompt = f"""
You are an AI property advisor at RealEstateCo.

Use this market context:
{context}

User question: "{query}"

Guidelines:
- Keep response short (2–3 sentences)
- End with: "Shall I prepare a detailed report?"
- Be professional and friendly
"""
        model = genai.GenerativeModel("gemini-2.5-flash")
        return model.generate_content(prompt).text.strip()

    def _generate_company_response(self, query: str) -> str:
        """Generate answers only from company profile"""
        if not self.company_profile.strip():
            return "Sorry, I couldn't load the company profile. Would you like me to connect you with an advisor?"

        prompt = f"""
You are an AI assistant at RealEstateCo.

Use ONLY this profile:
{self.company_profile}

User question: "{query}"

Guidelines:
- Max 3 sentences
- Include founder, address, contact if relevant
- End with: "If you'd like, I can arrange a call with our senior advisors."
- Never invent information
"""
        model = genai.GenerativeModel("gemini-2.5-flash")
        return model.generate_content(prompt).text.strip()


# ✅ Quick test
if __name__ == "__main__":
    rag = RealEstateRAG()
    test_queries = [
        "Who founded RealEstateCo and what services do you offer?",
        "What is the ROI trend for villas in Dubai?",
        "Explain the Dubai residential sale index trend",
    ]
    for q in test_queries:
        print(f"\n❓ {q}")
        print("📤", rag.query_knowledge_base(q))
