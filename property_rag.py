import os
from tqdm import tqdm
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import CSVLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

DATA_FILES = {
    "csv": [
        r"C:\Users\Keerthivasan R S\Videos\PYTHON PROGRAMS\FINAL RAG\dubai_property_index.csv",
        r"C:\Users\Keerthivasan R S\Videos\PYTHON PROGRAMS\FINAL RAG\summary_transactions.csv",
        r"C:\Users\Keerthivasan R S\Videos\PYTHON PROGRAMS\FINAL RAG\valuations1_rag.csv",
        r"C:\Users\Keerthivasan R S\Videos\PYTHON PROGRAMS\FINAL RAG\valuation2_rag.csv",
        r"C:\Users\Keerthivasan R S\Videos\PYTHON PROGRAMS\FINAL RAG\rents_rag.csv",
        r"C:\Users\Keerthivasan R S\Videos\PYTHON PROGRAMS\FINAL RAG\Residental_Sale_Index_rag.csv",
        r"C:\Users\Keerthivasan R S\Videos\PYTHON PROGRAMS\FINAL RAG\faq_questions.csv"
    ],
    "txt": [
        r"C:\Users\Keerthivasan R S\Videos\PYTHON PROGRAMS\FINAL RAG\about company.txt",
        r"C:\Users\Keerthivasan R S\Videos\PYTHON PROGRAMS\FINAL RAG\Dubai_Real_Estate_Market_Analysis.txt",
        r"C:\Users\Keerthivasan R S\Videos\PYTHON PROGRAMS\FINAL RAG\aboout the bot.txt"
    ]
}

COMPANY_PROFILE_PATH = r"C:\Users\Keerthivasan R S\Videos\PYTHON PROGRAMS\FINAL RAG\about company.txt"

class RealEstateRAG:
    """
    Smart context-aware RAG system for Baaz Landmark.
    Uses market data, property data, and official company profile to answer natural questions.
    """
    def __init__(self):
        self.INDEX_DIR = "baaz_landmark_faiss_index"
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.vectorstore = self._initialize_vectorstore()
        self.company_profile = self._load_company_profile()

    def _initialize_vectorstore(self):
        if os.path.exists(self.INDEX_DIR):
            print(f"✅ Loading existing FAISS index from {self.INDEX_DIR}")
            return FAISS.load_local(self.INDEX_DIR, self.embeddings, allow_dangerous_deserialization=True)

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
        docs = []
        docs.extend(self._load_csvs(DATA_FILES["csv"]))
        docs.extend(self._load_txts(DATA_FILES["txt"]))
        return docs

    def _load_csvs(self, filepaths):
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
        if os.path.exists(COMPANY_PROFILE_PATH):
            try:
                return TextLoader(COMPANY_PROFILE_PATH, encoding="utf-8").load()[0].page_content
            except Exception as e:
                print(f"❌ Could not load company profile: {e}")
                return ""
        else:
            print("⚠️ Company profile not found.")
            return ""

    def query_knowledge_base(self, query: str, k: int = 2):
        """
        Smart RAG query: brand questions go to company profile;
        else search vectorstore & answer.
        """
        query_lower = query.lower()
        if any(keyword in query_lower for keyword in ["company", "baaz", "founder", "ceo", "head office", "who are you"]):
            return self._generate_company_response(query)

        results = self.vectorstore.similarity_search(query, k=k)
        context = "\n\n".join([doc.page_content for doc in results])

        if not context.strip():
            return "Sorry, I couldn't find relevant data now. But I can connect you to a senior advisor!"

        prompt = f"""
You are Alexa, an AI advisor at Baaz Landmark.

Use this market context:
{context}

User question:
"{query}"

✅ Keep it short (2–3 sentences).
✅ Suggest next step: "Shall I prepare a detailed report?"
✅ Sound friendly, professional.
"""
        model = genai.GenerativeModel("gemini-2.5-flash")
        return model.generate_content(prompt).text.strip()

    def _generate_company_response(self, query: str):
        if not self.company_profile.strip():
            return "Sorry, I couldn't load our company profile — but I can connect you to a senior advisor."

        prompt = f"""
You are Alexa at Baaz Landmark.

Use ONLY this profile:
{self.company_profile}

User question: "{query}"

✅ Max 3 sentences.
✅ Include founder, address, contact if asked.
✅ End with: "If you'd like, I can arrange a call with our senior advisors."
✅ Never invent info.
"""
        model = genai.GenerativeModel("gemini-2.5-flash")
        return model.generate_content(prompt).text.strip()

# ✅ Quick test
if __name__ == "__main__":
    rag = RealEstateRAG()
    test_queries = [
        "Who founded Baaz Landmark and what services do you offer?",
        "What is the ROI trend for villas in Dubai?",
        "Explain the Dubai residential sale index trend",
    ]
    for q in test_queries:
        print(f"\n❓ {q}")
        print("📤", rag.query_knowledge_base(q))
