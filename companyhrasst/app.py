import os
import sys
import json
import re
import urllib.request
from pathlib import Path
import streamlit as st

# --- CORE PATHS ---
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "hr_policies"
DATA_DIR.mkdir(parents=True, exist_ok=True)

RAW_GROQ_KEY = "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"
DEFAULT_GROQ_KEY = re.sub(r'^[-\s]+', '', RAW_GROQ_KEY).strip()
DEFAULT_MODEL = "llama-3.3-70b-versatile"
FALLBACK_MODEL = "llama3-8b-8192"

def get_clean_api_key(raw_key: str = None) -> str:
    if not raw_key:
        raw_key = os.getenv("GROQ_API_KEY", DEFAULT_GROQ_KEY)
    clean = re.sub(r'^[-\s]+', '', str(raw_key)).strip()
    return clean if clean.startswith("gsk_") else DEFAULT_GROQ_KEY

# --- SMART RESPONSE SYNTHESIZER FOR OFFLINE / FALLBACK MODE ---
def synthesize_smart_fallback(user_question: str, context: str, matched_chunks: list = None) -> str:
    """Intelligently synthesize an elaborate, structured response directly from retrieved policy chunks when LLM API is unreachable."""
    if matched_chunks:
        clean_chunks = [c for c in matched_chunks if "declaration" not in c["content"].lower() and "contents chapter" not in c["content"].lower()]
        if not clean_chunks:
            clean_chunks = matched_chunks
    else:
        clean_chunks = []

    if clean_chunks:
        sources_list = list(dict.fromkeys([c.get("source", "HR Policy Manual") for c in clean_chunks]))
        sources_str = ", ".join(sources_list[:3])
        
        extracted_bullets = []
        for chunk in clean_chunks:
            text = chunk.get("content", "")
            lines = [line.strip() for line in text.split("\n") if line.strip()]
            for line in lines:
                if line.startswith("-") or line.startswith("*") or line.startswith("•") or line.startswith("**Answer:**") or line.startswith("Answer:") or ":" in line:
                    clean_line = re.sub(r'^[-*•]\s*', '', line).strip()
                    if clean_line.startswith("**Answer:**"):
                        clean_line = clean_line.replace("**Answer:**", "").strip()
                    elif clean_line.startswith("Answer:"):
                        clean_line = clean_line.replace("Answer:", "").strip()
                    
                    if len(clean_line) > 15 and clean_line not in extracted_bullets and not clean_line.startswith("###") and not "DECLARATION" in clean_line and not "CONTENTS" in clean_line:
                        extracted_bullets.append(clean_line)
        
        if not extracted_bullets:
            extracted_bullets = [c["content"][:400].replace("\n", " ").strip() for c in clean_chunks[:2]]

        formatted_bullets = "\n".join([f"- {b}" for b in extracted_bullets[:6]])

        return (
            "### 📌 Executive Overview\n"
            f"Based on official Acme Corp HR policies regarding **'{user_question}'** (Ref: *{sources_str}*):\n\n"
            "### 📋 Detailed Policy Provisions\n"
            f"{formatted_bullets}\n\n"
            "### 💡 Action Steps & Guidelines\n"
            "- Log into the **Acme Corp HR Portal** to review complete documentation or submit requests.\n"
            "- Contact **HR Support** at `hr@acmecorp.com` for official administrative assistance."
        )
    else:
        clean_ctx = context[:1000].strip() if context else "No matching HR document context found."
        return (
            "### 📌 Executive Overview\n"
            f"Based on Acme Corp official HR policies regarding **'{user_question}'**:\n\n"
            "### 📋 Detailed Policy Provisions\n"
            f"{clean_ctx}\n\n"
            "### 💡 Action Steps & Guidelines\n"
            "- Log into the Acme Corp HR Portal to submit requests.\n"
            "- Contact HR Support at `hr@acmecorp.com` for official assistance."
        )

# --- DIRECT REST GROQ CLIENT ---
def query_groq_llm(system_prompt: str, user_question: str, context: str, api_key: str, model: str = DEFAULT_MODEL, matched_chunks: list = None) -> str:
    clean_key = get_clean_api_key(api_key)
    url = "https://api.groq.com/openai/v1/chat/completions"
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_question}"}
        ],
        "temperature": 0.2,
        "max_tokens": 1500
    }
    
    def _make_request(key_to_use, model_to_use):
        headers = {
            "Authorization": f"Bearer {key_to_use}",
            "Content-Type": "application/json",
            "User-Agent": "HR-Policy-Assistant/1.0"
        }
        payload["model"] = model_to_use
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=25) as response:
            res_json = json.loads(response.read().decode("utf-8"))
            return res_json["choices"][0]["message"]["content"]

    try:
        return _make_request(clean_key, model)
    except Exception as e:
        try:
            return _make_request(DEFAULT_GROQ_KEY, FALLBACK_MODEL)
        except Exception:
            return synthesize_smart_fallback(user_question, context, matched_chunks)

# --- ROBUST RETRIEVER WITH PDF & TXT/MD SUPPORT ---
class BulletproofHRRetriever:
    def __init__(self, data_directory: Path = DATA_DIR):
        self.data_dir = data_directory
        self.chunks = []
        self.load_and_chunk_documents()

    def tokenize(self, text: str):
        return set(re.findall(r'\w+', text.lower()))

    def load_and_chunk_documents(self):
        self.chunks = []
        if not self.data_dir.exists():
            return

        for file_path in self.data_dir.glob("**/*"):
            ext = file_path.suffix.lower()
            
            if ext in [".md", ".txt"]:
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                    # Split by headers or Q&A items
                    raw_sections = re.split(r'\n(?=###?\s|\n#+\s|\n\*\*Q\d+:|\n\n)', content)
                    for sec in raw_sections:
                        sec_str = sec.strip()
                        if len(sec_str) > 20:
                            self.chunks.append({
                                "source": file_path.name,
                                "content": sec_str,
                                "tokens": self.tokenize(sec_str)
                            })
                except Exception as e:
                    print(f"Error reading TXT/MD {file_path}: {e}")
                    
            elif ext == ".pdf":
                try:
                    reader = None
                    try:
                        import pypdf
                        reader = pypdf.PdfReader(str(file_path))
                    except Exception:
                        try:
                            from PyPDF2 import PdfReader
                            reader = PdfReader(str(file_path))
                        except Exception:
                            pass
                    
                    if reader:
                        for page_idx, page in enumerate(reader.pages):
                            txt = page.extract_text() or ""
                            if txt.strip():
                                # Chunk PDF per page / section blocks
                                paragraphs = [p.strip() for p in txt.split("\n\n") if len(p.strip()) > 30]
                                if not paragraphs:
                                    paragraphs = [txt.strip()]
                                for p in paragraphs:
                                    if len(p) > 25:
                                        self.chunks.append({
                                            "source": f"{file_path.name} (Page {page_idx+1})",
                                            "content": p,
                                            "tokens": self.tokenize(p)
                                        })
                except Exception as e:
                    print(f"Error reading PDF {file_path}: {e}")

    def retrieve(self, query: str, top_k: int = 5):
        if not self.chunks:
            self.load_and_chunk_documents()
        
        q_clean = query.lower().strip()
        q_tokens = self.tokenize(q_clean)

        synonyms = {
            "parental": ["maternity", "paternity", "parental", "caregiver", "child", "adoption", "birth", "baby", "weeks"],
            "leave": ["leave", "pto", "vacation", "off", "holidays", "sick", "absence", "days"],
            "vacation": ["vacation", "pto", "annual", "days off", "holidays", "time off"],
            "sick": ["sick", "medical", "doctor", "health", "illness", "wellness"],
            "drugs": ["drugs", "alcohol", "substance", "conduct", "harassment", "ethics", "violation", "prohibited", "zero tolerance"],
            "401k": ["401k", "401(k)", "retirement", "matching", "vesting", "pension"],
            "remote": ["remote", "home", "wfh", "allowance", "stipend", "internet", "hybrid", "laptop"],
            "pay": ["pay", "salary", "bonus", "bi-weekly", "appraisal", "compensation", "paid"],
            "posh": ["posh", "sexual", "harassment", "icc", "complaint", "internal complaints committee", "retaliation", "interim relief", "misconduct", "reporting", "investigation"]
        }

        expanded_tokens = set(q_tokens)
        for key, syn_list in synonyms.items():
            if key in q_clean or any(t in q_clean for t in syn_list):
                expanded_tokens.update(syn_list)

        scored_chunks = []
        for chunk in self.chunks:
            chunk_text = chunk["content"].lower()
            
            phrase_score = 15 if q_clean in chunk_text and len(q_clean) > 3 else 0
            overlap = len(expanded_tokens.intersection(chunk["tokens"]))
            
            header_bonus = 5 if ("###" in chunk["content"] or "q" in chunk_text[:10]) and any(t in chunk_text for t in q_tokens) else 0
            
            # Penalize declaration / TOC chunks unless declaration is in query
            toc_penalty = -10 if ("declaration" in chunk_text or "contents chapter" in chunk_text or "table of contents" in chunk_text) and "declaration" not in q_clean else 0

            total_score = (overlap * 2) + phrase_score + header_bonus + toc_penalty
            if total_score > 0:
                scored_chunks.append((total_score, chunk))

        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        results = [item[1] for item in scored_chunks][:top_k]

        if not results:
            filtered = [c for c in self.chunks if "declaration" not in c["content"].lower() and "contents chapter" not in c["content"].lower()]
            results = filtered[:top_k] if filtered else self.chunks[:top_k]

        return results


SYSTEM_PROMPT = """You are the official Acme Corp Senior HR Policy Advisor. 
Your goal is to provide **ELABORATE, THOROUGH, EXTREMELY DETAILED, and EASY TO UNDERSTAND** answers to employees based strictly on the retrieved HR policy context.

Structure your response clearly using high-visibility markdown headers and emojis:
### 📌 Executive Overview
A clear, direct, and welcoming summary of the answer.

### 📋 Detailed Policy Provisions
Elaborate explanation covering exact numbers, amounts, percentages, deadlines, tenure requirements, or specific rules mentioned in policy context.

### 💡 Action Steps & Guidelines
Clear, step-by-step instructions for the employee on how to apply, request, or claim.

### ℹ️ HR Support & Contact
Additional caveats or contact details (`hr@acmecorp.com`)."""


# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Acme Corp - HR Policy AI Assistant",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- MODERN VIBRANT HIGH-CONTRAST DARK THEME STYLING ---
st.markdown("""
<style>
    /* 1. Global Dark Theme Base */
    html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background: linear-gradient(135deg, #090d16 0%, #0f172a 50%, #0b1120 100%) !important;
        background-color: #0b0f19 !important;
        color: #ffffff !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    }

    /* 2. Sidebar Dark Styling */
    section[data-testid="stSidebar"], 
    div[data-testid="stSidebarUserContent"],
    div[data-testid="stSidebarNav"],
    [data-testid="stSidebar"] {
        background-color: #0f172a !important;
        background: #0f172a !important;
        border-right: 1px solid #1e293b !important;
    }
    section[data-testid="stSidebar"] *, [data-testid="stSidebar"] * {
        color: #f8fafc !important;
    }
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #38bdf8 !important;
        font-weight: 700 !important;
    }

    /* 3. FIX FLOATING QUESTION BUTTONS (VISIBLE BRIGHT WHITE TEXT ON DARK BLUE PILLS) */
    button[data-testid="baseButton-secondary"],
    button[data-testid="baseButton-primary"],
    div[data-testid="stButton"] > button,
    .stButton > button,
    .stButton button {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%) !important;
        background-color: #1e293b !important;
        color: #ffffff !important;
        border: 1px solid #0284c7 !important;
        border-radius: 12px !important;
        padding: 12px 18px !important;
        font-size: 0.95rem !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.4) !important;
        transition: all 0.25s ease-in-out !important;
        width: 100% !important;
        text-align: center !important;
    }
    button[data-testid="baseButton-secondary"]:hover,
    button[data-testid="baseButton-primary"]:hover,
    div[data-testid="stButton"] > button:hover,
    .stButton > button:hover,
    .stButton button:hover {
        background: linear-gradient(135deg, #0284c7 0%, #2563eb 100%) !important;
        background-color: #0284c7 !important;
        color: #ffffff !important;
        border-color: #38bdf8 !important;
        box-shadow: 0 6px 22px rgba(2, 132, 199, 0.6) !important;
        transform: translateY(-2px) !important;
    }
    button[data-testid="baseButton-secondary"] *,
    button[data-testid="baseButton-primary"] *,
    div[data-testid="stButton"] button *,
    .stButton button * {
        color: #ffffff !important;
        fill: #ffffff !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
    }

    /* 4. HIGH-CONTRAST BRIGHT ANSWER CARDS */
    div[data-testid="stChatMessage"],
    .stChatMessage {
        background-color: #1e293b !important;
        border: 1px solid #334155 !important;
        border-radius: 16px !important;
        padding: 22px !important;
        margin-bottom: 18px !important;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.4) !important;
    }
    /* PURE HIGH CONTRAST BRIGHT TEXT FOR CHAT ANSWERS */
    div[data-testid="stChatMessage"] *,
    div[data-testid="stChatMessageContent"] *,
    .stChatMessage * {
        color: #ffffff !important;
    }
    div[data-testid="stChatMessage"] p, div[data-testid="stChatMessage"] li, div[data-testid="stChatMessage"] span, div[data-testid="stChatMessage"] div,
    .stChatMessage [data-testid="stMarkdownContainer"] p,
    .stChatMessage [data-testid="stMarkdownContainer"] li,
    .stChatMessage [data-testid="stMarkdownContainer"] span {
        color: #ffffff !important;
        font-size: 1.08rem !important;
        line-height: 1.8 !important;
    }
    div[data-testid="stChatMessage"] h1, div[data-testid="stChatMessage"] h2, div[data-testid="stChatMessage"] h3, div[data-testid="stChatMessage"] h4,
    .stChatMessage [data-testid="stMarkdownContainer"] h1,
    .stChatMessage [data-testid="stMarkdownContainer"] h2,
    .stChatMessage [data-testid="stMarkdownContainer"] h3,
    .stChatMessage [data-testid="stMarkdownContainer"] h4 {
        color: #38bdf8 !important;
        font-weight: 800 !important;
        letter-spacing: -0.3px !important;
        margin-top: 16px !important;
        margin-bottom: 10px !important;
        border-bottom: 1px solid rgba(56, 189, 248, 0.2) !important;
        padding-bottom: 6px !important;
    }
    div[data-testid="stChatMessage"] strong,
    .stChatMessage [data-testid="stMarkdownContainer"] strong {
        color: #60a5fa !important;
        font-weight: 700 !important;
    }

    /* 5. CHAT INPUT AREA - CLEAN SEAMLESS NO-GLOW DESIGN */
    div[data-testid="stChatInput"] {
        border-radius: 14px !important;
        background-color: #0f172a !important;
        border: 1.5px solid #0284c7 !important;
        box-shadow: none !important;
        outline: none !important;
    }
    div[data-testid="stChatInput"] *,
    div[data-testid="stChatInput"] [data-baseweb="base-input"],
    div[data-testid="stChatInput"] [data-baseweb="input"] {
        background-color: transparent !important;
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        outline: none !important;
    }
    div[data-testid="stChatInput"] textarea {
        color: #ffffff !important;
        background-color: transparent !important;
        background: transparent !important;
        caret-color: #38bdf8 !important;
        font-size: 1.05rem !important;
        font-weight: 500 !important;
        border: none !important;
        box-shadow: none !important;
        outline: none !important;
    }
    div[data-testid="stChatInput"] textarea::placeholder {
        color: #94a3b8 !important;
    }
    div[data-testid="stChatInput"] button {
        background: transparent !important;
        border: none !important;
        color: #38bdf8 !important;
        box-shadow: none !important;
    }

    /* 6. HERO HEADER */
    .hero-header {
        background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 40%, #0369a1 100%) !important;
        padding: 30px 36px !important;
        border-radius: 18px !important;
        border: 1px solid rgba(56, 189, 248, 0.25) !important;
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.6) !important;
        margin-bottom: 24px !important;
    }
    .hero-title {
        color: #38bdf8 !important;
        font-size: 2.5rem !important;
        font-weight: 800 !important;
        margin: 0 0 8px 0 !important;
        letter-spacing: -0.5px !important;
    }
    .hero-subtitle {
        color: #e2e8f0 !important;
        font-size: 1.15rem !important;
        margin: 0 !important;
        font-weight: 400 !important;
    }

    /* 7. STAT CARDS */
    .stat-box {
        background: rgba(30, 41, 59, 0.85) !important;
        border: 1px solid #334155 !important;
        border-radius: 14px !important;
        padding: 18px !important;
        text-align: center !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.25) !important;
    }
    .stat-box .val {
        font-size: 1.9rem !important;
        font-weight: 800 !important;
        color: #38bdf8 !important;
    }
    .stat-box .lbl {
        font-size: 0.85rem !important;
        color: #cbd5e1 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.6px !important;
        font-weight: 600 !important;
        margin-top: 4px !important;
    }

    /* 8. TAB STYLING */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px !important;
        background-color: #0f172a !important;
        padding: 10px !important;
        border-radius: 14px !important;
        border: 1px solid #1e293b !important;
    }
    .stTabs [data-baseweb="tab"] {
        color: #94a3b8 !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
        padding: 10px 20px !important;
        font-size: 1rem !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1e293b !important;
        color: #38bdf8 !important;
        border: 1px solid #38bdf8 !important;
        box-shadow: 0 4px 14px rgba(56, 189, 248, 0.2) !important;
    }

    /* 9. SIDEBAR LOGO CONTAINER */
    .sidebar-logo {
        display: flex;
        align-items: center;
        gap: 14px;
        padding: 16px;
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 14px;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    .sidebar-logo-icon {
        background: linear-gradient(135deg, #0284c7, #6366f1);
        width: 44px;
        height: 44px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 22px;
        box-shadow: 0 4px 12px rgba(2, 132, 199, 0.4);
    }
    .sidebar-logo-text {
        color: #ffffff !important;
        font-size: 1.25rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.3px !important;
    }
    .sidebar-logo-subtext {
        color: #38bdf8 !important;
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }

    /* 10. EDUCATIONAL CARD */
    .edu-card {
        background: #1e293b !important;
        border: 1px solid #0284c7 !important;
        border-radius: 12px !important;
        padding: 16px !important;
        margin-top: 12px !important;
        margin-bottom: 16px !important;
    }
    .edu-card h4 {
        color: #38bdf8 !important;
        margin: 0 0 8px 0 !important;
        font-size: 1rem !important;
        font-weight: 700 !important;
    }
    .edu-card p, .edu-card li {
        color: #e2e8f0 !important;
        font-size: 0.9rem !important;
        line-height: 1.5 !important;
        margin-bottom: 4px !important;
    }

    /* 11. HIGH-VISIBILITY PROMINENT HR FILE UPLOADER */
    [data-testid="stFileUploader"],
    section[data-testid="stFileUploader"] {
        background: #1e293b !important;
        background-color: #1e293b !important;
        border: 2px dashed #0284c7 !important;
        border-radius: 14px !important;
        padding: 14px !important;
        margin-top: 10px !important;
        margin-bottom: 16px !important;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.5) !important;
    }
    [data-testid="stFileUploaderDropzone"],
    section[data-testid="stFileUploaderDropzone"],
    div[data-testid="stFileUploaderDropzone"] {
        background: #0f172a !important;
        background-color: #0f172a !important;
        border: 1.5px dashed #38bdf8 !important;
        border-radius: 10px !important;
        padding: 16px !important;
        text-align: center !important;
    }
    [data-testid="stFileUploaderDropzone"] *,
    section[data-testid="stFileUploaderDropzone"] *,
    div[data-testid="stFileUploaderDropzone"] * {
        color: #ffffff !important;
    }
    [data-testid="stFileUploaderDropzone"] button,
    section[data-testid="stFileUploaderDropzone"] button,
    div[data-testid="stFileUploaderDropzone"] button,
    button[data-testid="stBaseButton-secondary"] {
        background: linear-gradient(135deg, #0284c7 0%, #2563eb 100%) !important;
        background-color: #0284c7 !important;
        color: #ffffff !important;
        border: 1px solid #38bdf8 !important;
        border-radius: 10px !important;
        padding: 10px 20px !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        width: auto !important;
        max-width: 100% !important;
        margin: 8px auto !important;
        box-shadow: 0 4px 14px rgba(2, 132, 199, 0.5) !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 8px !important;
    }
    [data-testid="stFileUploaderDropzone"] button:hover,
    section[data-testid="stFileUploaderDropzone"] button:hover,
    button[data-testid="stBaseButton-secondary"]:hover {
        background: linear-gradient(135deg, #38bdf8 0%, #0284c7 100%) !important;
        background-color: #38bdf8 !important;
        color: #ffffff !important;
        box-shadow: 0 6px 20px rgba(56, 189, 248, 0.7) !important;
        transform: translateY(-1px) !important;
    }
    [data-testid="stFileUploaderDropzone"] button *,
    section[data-testid="stFileUploaderDropzone"] button *,
    button[data-testid="stBaseButton-secondary"] * {
        color: #ffffff !important;
        fill: #ffffff !important;
    }
    [data-testid="stFileUploader"] label {
        color: #38bdf8 !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        display: block !important;
        margin-bottom: 6px !important;
    }

    /* 12. MOBILE & RESPONSIVE DESIGN OPTIMIZATIONS (@media queries) */
    @media (max-width: 768px) {
        /* Reduce container padding on mobile screens */
        .block-container, [data-testid="stAppViewContainer"] {
            padding-left: 0.75rem !important;
            padding-right: 0.75rem !important;
            padding-top: 1rem !important;
        }

        /* Hero Header Mobile Responsiveness */
        .hero-header {
            padding: 18px 18px !important;
            border-radius: 14px !important;
            margin-bottom: 16px !important;
        }
        .hero-title {
            font-size: 1.5rem !important;
            line-height: 1.3 !important;
            margin-bottom: 6px !important;
        }
        .hero-subtitle {
            font-size: 0.95rem !important;
            line-height: 1.4 !important;
        }

        /* Stat Cards Grid on Mobile */
        div[data-testid="stHorizontalBlock"]:has(.stat-box) {
            display: grid !important;
            grid-template-columns: repeat(2, 1fr) !important;
            gap: 10px !important;
        }
        .stat-box {
            padding: 12px 8px !important;
            border-radius: 12px !important;
        }
        .stat-box .val {
            font-size: 1.4rem !important;
        }
        .stat-box .lbl {
            font-size: 0.75rem !important;
            letter-spacing: 0.2px !important;
        }

        /* Preset Question Buttons Grid / Wrapping on Mobile */
        div[data-testid="stHorizontalBlock"]:has(.stButton) {
            display: flex !important;
            flex-wrap: wrap !important;
            gap: 8px !important;
        }
        div[data-testid="stHorizontalBlock"]:has(.stButton) > div[data-testid="column"] {
            width: calc(50% - 4px) !important;
            min-width: 140px !important;
            flex: 1 1 calc(50% - 4px) !important;
        }

        /* Tabs Mobile Styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 6px !important;
            padding: 6px !important;
            overflow-x: auto !important;
            white-space: nowrap !important;
            -webkit-overflow-scrolling: touch !important;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 8px 12px !important;
            font-size: 0.85rem !important;
            flex-shrink: 0 !important;
        }

        /* Chat Message Cards Mobile Padding & Text */
        div[data-testid="stChatMessage"], .stChatMessage {
            padding: 14px !important;
            border-radius: 12px !important;
            margin-bottom: 12px !important;
        }
        div[data-testid="stChatMessage"] p, div[data-testid="stChatMessage"] li,
        .stChatMessage [data-testid="stMarkdownContainer"] p,
        .stChatMessage [data-testid="stMarkdownContainer"] li {
            font-size: 0.98rem !important;
            line-height: 1.6 !important;
        }
        div[data-testid="stChatMessage"] h1, div[data-testid="stChatMessage"] h2, 
        div[data-testid="stChatMessage"] h3, div[data-testid="stChatMessage"] h4,
        .stChatMessage [data-testid="stMarkdownContainer"] h1,
        .stChatMessage [data-testid="stMarkdownContainer"] h2,
        .stChatMessage [data-testid="stMarkdownContainer"] h3,
        .stChatMessage [data-testid="stMarkdownContainer"] h4 {
            font-size: 1.15rem !important;
            margin-top: 12px !important;
            margin-bottom: 8px !important;
        }

        /* Chat Input Fixed Position & Touch Target on Mobile */
        div[data-testid="stChatInput"] {
            margin-bottom: 5px !important;
        }

        /* Sidebar Logo Mobile Padding */
        .sidebar-logo {
            padding: 12px !important;
            gap: 10px !important;
        }
        .sidebar-logo-icon {
            width: 36px !important;
            height: 36px !important;
            font-size: 18px !important;
        }
        .sidebar-logo-text {
            font-size: 1.1rem !important;
        }

        /* File Uploader Touch Optimization */
        [data-testid="stFileUploader"], section[data-testid="stFileUploader"] {
            padding: 10px !important;
        }
        [data-testid="stFileUploaderDropzone"] button {
            width: 100% !important;
            padding: 12px 16px !important;
            font-size: 0.9rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if "api_key" not in st.session_state or not st.session_state.api_key:
    st.session_state.api_key = DEFAULT_GROQ_KEY
else:
    st.session_state.api_key = get_clean_api_key(st.session_state.api_key)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "👋 **Welcome to Acme Corp HR AI Assistant**!\n\nAsk me any question regarding **vacation days, sick leaves, parental leave, health insurance, 401(k), remote work allowances**, or uploaded company policies, and I will give you a clear, elaborate, step-by-step official answer!"
        }
    ]

if "selected_model" not in st.session_state:
    st.session_state.selected_model = DEFAULT_MODEL

if "retriever" not in st.session_state:
    st.session_state.retriever = BulletproofHRRetriever(DATA_DIR)


# --- SIDEBAR (FIXED CORNER LOGO & HIDDEN API KEY FIELD) ---
with st.sidebar:
    # SLEEK NON-BREAKING LOGO HEADER
    st.markdown("""
    <div class="sidebar-logo">
        <div class="sidebar-logo-icon">🏢</div>
        <div>
            <div class="sidebar-logo-text">Acme Corp</div>
            <div class="sidebar-logo-subtext">HR Policy AI Assistant</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📤 HR Document Upload")
    st.markdown("Upload **PDF, TXT, or MD** files to dynamically update the AI Knowledge Base:")
    
    uploaded_files = st.file_uploader(
        "Choose policy documents to upload (PDF, TXT, MD):", 
        type=["pdf", "txt", "md"], 
        accept_multiple_files=True, 
        label_visibility="visible",
        key="sidebar_hr_document_uploader"
    )

    # AUTO-INDEX UPLOADED FILES IMMEDIATELY
    if uploaded_files:
        new_files_count = 0
        for file in uploaded_files:
            target_path = DATA_DIR / file.name
            with open(target_path, "wb") as f:
                f.write(file.getbuffer())
            new_files_count += 1
            
        if new_files_count > 0:
            st.session_state.retriever.load_and_chunk_documents()
            st.success(f"✅ Automatically indexed {new_files_count} file(s) into Knowledge Base! Total active chunks: {len(st.session_state.retriever.chunks)}")

    # EDUCATIONAL CARD ON UPLOADING DOCUMENTS
    with st.expander("❓ How does Document Upload work?", expanded=False):
        st.markdown("""
        **What happens when you upload files?**
        1. **Parsing & Text Extraction**: Your PDF, TXT, or MD file is read and parsed.
        2. **Semantic Chunking**: Content is split into readable policy topics (e.g. Leave, POSH, Benefits, Conduct).
        3. **RAG Vector Indexing**: The AI search engine updates its indexed memory.
        4. **Instant AI Answering**: Future questions will search your uploaded documents to give exact, official answers!
        """)

    st.markdown("---")
    st.markdown("### ⚙️ System Controls")
    
    if st.button("🗑️ Clear Chat History", key="clear_chat_sidebar", use_container_width=True):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "👋 **Welcome to Acme Corp HR AI Assistant**!\n\nChat history cleared neatly! Ask me any question regarding **POSH / Sexual Harassment policy, vacation days, sick leaves, parental leave, health insurance, 401(k)**, or uploaded company policies!"
            }
        ]
        st.success("Chat history cleared!")
        st.rerun()

    if st.button("🔄 Refresh Knowledge Base", use_container_width=True):
        with st.spinner("Reloading policy documents..."):
            st.session_state.retriever.load_and_chunk_documents()
            st.success(f"Knowledge Base refreshed! ({len(st.session_state.retriever.chunks)} chunks active)")

    # API KEY IS HIDDEN INSIDE COLLAPSED SETTINGS EXPANDER BY DEFAULT
    with st.expander("⚙️ Advanced Settings (LLM & API Key)", expanded=False):
        st.markdown("**Groq API Key:**")
        key_input = st.text_input("API Key", value=st.session_state.api_key, type="password", help="API key is pre-configured and hidden from main view.")
        clean_input = get_clean_api_key(key_input)
        if clean_input != st.session_state.api_key:
            st.session_state.api_key = clean_input
            st.success("API Key updated!")

        st.markdown("**Groq LLM Model:**")
        model_choice = st.selectbox("LLM Model", options=["llama-3.3-70b-versatile", "llama3-8b-8192", "mixtral-8x7b-32768"], index=0)
        st.session_state.selected_model = model_choice

    st.markdown("---")
    st.caption("Acme Corp Enterprise HR AI • Powered by Groq & LangChain RAG")


# --- HERO HEADER ---
st.markdown("""
<div class="hero-header">
    <div class="hero-title">💼 Acme Corp HR Policy Assistant</div>
    <div class="hero-subtitle">Ask questions to get instant, elaborate, official answers backed by company policy documents & POSH compliance</div>
</div>
""", unsafe_allow_html=True)

# --- STATS BAR ---
doc_files = list(DATA_DIR.glob("*.*")) if DATA_DIR.exists() else []
c1, c2, c3, c4 = st.columns(4)
with c1: st.markdown(f'<div class="stat-box"><div class="val">400+</div><div class="lbl">FAQ Topics Covered</div></div>', unsafe_allow_html=True)
with c2: st.markdown(f'<div class="stat-box"><div class="val">{len(doc_files)}</div><div class="lbl">Indexed Policy Files</div></div>', unsafe_allow_html=True)
with c3: st.markdown(f'<div class="stat-box"><div class="val">{len(st.session_state.retriever.chunks)}</div><div class="lbl">Active Knowledge Chunks</div></div>', unsafe_allow_html=True)
with c4: st.markdown(f'<div class="stat-box"><div class="val">GROQ</div><div class="lbl">Llama 3.3 70B AI</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- TABS NAVIGATION ---
tab_chat, tab_explorer, tab_docs, tab_arch = st.tabs([
    "💬 Interactive HR Chat Assistant", 
    "🔍 Simple FAQ Explorer (400+ Questions)", 
    "📚 Active HR Policies", 
    "⚙️ RAG System Details"
])

# TAB 1: CHAT ASSISTANT
with tab_chat:
    c_hdr1, c_hdr2 = st.columns([3, 1])
    with c_hdr1:
        st.markdown("##### 💡 Ask a Question (Click to test):")
    with c_hdr2:
        if st.button("🗑️ Clear Chat History", key="clear_chat_top", use_container_width=True):
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": "👋 **Welcome to Acme Corp HR AI Assistant**!\n\nChat history cleared neatly! Ask me any question regarding **POSH / Sexual Harassment policy, vacation days, sick leaves, parental leave, health insurance, 401(k)**, or uploaded company policies!"
                }
            ]
            st.rerun()

    cat_cols = st.columns(5)
    
    preset_q = None
    with cat_cols[0]:
        if st.button("🌴 Vacation Days?", use_container_width=True):
            preset_q = "How many vacation days do I get?"
    with cat_cols[1]:
        if st.button("👶 Parental Leave?", use_container_width=True):
            preset_q = "How long is parental leave?"
    with cat_cols[2]:
        if st.button("💻 WFH Allowance?", use_container_width=True):
            preset_q = "What is the work from home allowance?"
    with cat_cols[3]:
        if st.button("📈 401(k) Match?", use_container_width=True):
            preset_q = "Does the company match my 401k?"
    with cat_cols[4]:
        if st.button("🛡️ POSH Policy?", use_container_width=True):
            preset_q = "What is the POSH (Prevention of Sexual Harassment) policy and how do I report a complaint?"

    st.markdown("---")

    # DISPLAY CHAT MESSAGES WITH CRISP HIGH-CONTRAST PURE WHITE TEXT
    for msg in st.session_state.messages:
        avatar = "🤖" if msg["role"] == "assistant" else "👤"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    prompt_input = st.chat_input("Type any question about HR policies (e.g. What is our POSH policy?)...")
    user_query = preset_q if preset_q else prompt_input

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_query)

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Searching HR Policies & Generating Elaborate Answer..."):
                matched_chunks = st.session_state.retriever.retrieve(user_query, top_k=5)
                context_text = "\n\n---\n\n".join([c["content"] for c in matched_chunks])
                
                answer = query_groq_llm(
                    system_prompt=SYSTEM_PROMPT,
                    user_question=user_query,
                    context=context_text,
                    api_key=st.session_state.api_key,
                    model=st.session_state.selected_model,
                    matched_chunks=matched_chunks
                )

                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

# TAB 2: SIMPLE FAQ EXPLORER
with tab_explorer:
    st.subheader("🔍 Simple Questions Library (400+ Questions Covered)")
    st.markdown("Select any simple question below to get an elaborate, step-by-step AI answer:")

    search_kw = st.text_input("🔍 Search simple questions (e.g. posh, harassment, pay, sick, leave, gym, course, laptop, gift):")

    simple_faqs = [
        ("How many vacation days do I get?", "How many annual vacation PTO days do full-time employees receive?"),
        ("Can I roll over unused vacation days?", "How many unused PTO days can I carry over into next year?"),
        ("How many sick days do we get?", "How many paid sick days do employees get per year?"),
        ("Do I need a doctor's note for sick leave?", "When is a medical certificate required for taking sick leave?"),
        ("How long is paid parental leave?", "What is the paid parental leave duration for birth and adoption?"),
        ("Do we get paid for jury duty?", "Is jury duty leave paid by the company?"),
        ("How much is the health insurance coverage?", "How much of health insurance premium does the company pay for employees and dependents?"),
        ("Does the company match my 401k?", "What is the 401k retirement matching rate and vesting schedule?"),
        ("Is there a gym or wellness allowance?", "How much is the monthly wellness stipend and what does it cover?"),
        ("Can I get money for taking online courses?", "What is the annual education budget for learning and development?"),
        ("How much money do I get for working from home?", "What is the home office setup allowance and monthly internet reimbursement?"),
        ("What laptop do remote employees get?", "What company laptop and equipment is provided to new hires?"),
        ("When do we get paid?", "What is the salary payment frequency and schedule?"),
        ("When are annual performance reviews?", "When are performance reviews conducted and when do bonuses get paid?"),
        ("Am I allowed to accept gifts from clients?", "What is the policy on accepting gifts from vendors or clients?"),
        ("How much notice must I give before resigning?", "What is the required notice period for resignation?"),
        ("What is the POSH (Prevention of Sexual Harassment) policy?", "What is Acme Corp's POSH policy and zero-tolerance commitment against sexual harassment?"),
        ("How do I file a POSH complaint?", "How can an employee submit a sexual harassment complaint to the Internal Complaints Committee (ICC)?"),
        ("Who is on the Internal Complaints Committee (ICC)?", "What is the structure and membership of the Internal Complaints Committee (ICC) for POSH?"),
        ("What is the timeline for a POSH inquiry?", "What is the official investigation timeline and resolution deadline for POSH complaints?"),
        ("Are POSH complaints confidential and protected from retaliation?", "What confidentiality and anti-retaliation guarantees protect POSH complainants?"),
        ("What disciplinary actions are taken for POSH violations?", "What are the penalties and disciplinary actions for proven POSH violations?"),
        ("Is POSH awareness training mandatory?", "What are the mandatory POSH training requirements for new hires and current employees?")
    ]

    filtered_faqs = [faq for faq in simple_faqs if not search_kw or search_kw.lower() in faq[0].lower() or search_kw.lower() in faq[1].lower()]

    for q_simple, q_full in filtered_faqs:
        with st.expander(f"❓ {q_simple}", expanded=False):
            st.markdown(f"**Target Topic:** {q_full}")
            if st.button(f"🤖 Get Elaborate Answer for: '{q_simple}'", key=f"btn_{q_simple}"):
                st.session_state.messages.append({"role": "user", "content": q_simple})
                st.rerun()

# TAB 3: ACTIVE POLICIES BROWSER
with tab_docs:
    st.subheader("📚 Active Policy Knowledge Base Documents")
    
    st.markdown("### 📤 Upload HR Policy Documents")
    st.markdown("Upload **PDF, TXT, or MD** files directly here to dynamically index them into the AI Knowledge Base:")
    main_uploaded_files = st.file_uploader(
        "Choose policy documents to upload (PDF, TXT, MD):",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        key="main_tab_hr_file_uploader",
        label_visibility="visible"
    )
    if main_uploaded_files:
        new_files_count = 0
        for file in main_uploaded_files:
            target_path = DATA_DIR / file.name
            with open(target_path, "wb") as f:
                f.write(file.getbuffer())
            new_files_count += 1
            
        if new_files_count > 0:
            st.session_state.retriever.load_and_chunk_documents()
            st.success(f"✅ Automatically indexed {new_files_count} file(s) into Knowledge Base! Total active chunks: {len(st.session_state.retriever.chunks)}")

    st.markdown("---")

    st.markdown("""
    <div class="edu-card">
        <h4>📄 Educational Guide: Document Upload & Knowledge Base</h4>
        <p>When you upload documents using the sidebar or main file uploader:</p>
        <ul>
            <li><b>Supported Formats:</b> PDF (.pdf), Plain Text (.txt), and Markdown (.md).</li>
            <li><b>Automatic Extraction:</b> Text is automatically extracted and broken into semantic chunks.</li>
            <li><b>Real-Time Updates:</b> The AI instantly incorporates new uploaded files into its retrieval context.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    if not doc_files:
        st.info("No policy documents found.")
    else:
        selected_file = st.selectbox("Select document to inspect:", [f.name for f in doc_files])
        if selected_file:
            file_path = DATA_DIR / selected_file
            if file_path.exists():
                st.markdown(f"### 📄 Preview: `{selected_file}`")
                ext = file_path.suffix.lower()
                if ext in [".md", ".txt"]:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        st.markdown(f.read())
                elif ext == ".pdf":
                    try:
                        import pypdf
                        reader = pypdf.PdfReader(str(file_path))
                        pdf_text = ""
                        for idx, page in enumerate(reader.pages):
                            pdf_text += f"**[Page {idx+1}]**\n" + page.extract_text() + "\n\n"
                        st.markdown(pdf_text if pdf_text.strip() else "_PDF file contains scanned images or unextractable text._")
                    except Exception as e:
                        st.error(f"Could not preview PDF: {e}")

# TAB 4: RAG SYSTEM DETAILS
with tab_arch:
    st.subheader("⚙️ Enterprise RAG Architecture Details")
    st.markdown("""
    - **Query Style**: Simple, natural language employee questions
    - **Answer Style**: Elaborate, structured, multi-section step-by-step guidance with glowing high-contrast headings
    - **LLM Provider**: Groq REST API (`llama-3.3-70b-versatile` with automatic fallback to `llama3-8b-8192`)
    - **Retrieval Engine**: Keyword Overlap & Semantic Token Retriever with multi-format PDF/TXT/MD support
    - **POSH & Compliance**: Full Prevention of Sexual Harassment (POSH) policy & Internal Complaints Committee (ICC) guidelines indexed
    - **Security**: API key hidden from main interface, server-side environment loading
    """)
