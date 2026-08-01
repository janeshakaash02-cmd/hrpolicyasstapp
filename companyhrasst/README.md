# Acme Corp - HR Policy Assistant (RAG POC)

A modern, robust Retrieval-Augmented Generation (RAG) Proof of Concept (POC) built with **Python**, **LangChain**, **Groq Cloud API** (`llama-3.3-70b-versatile`), local HuggingFace embeddings (`all-MiniLM-L6-v2`), **FAISS Vector Store**, and an executive **Streamlit Web UI/UX**.

---

## 🚀 Features

- 💼 **Comprehensive HR Policies Knowledge Base**: Included realistic company policies covering PTO, Sick Leave, Parental Leave, Code of Conduct, Ethics, Health Insurance, 401(k), Performance Reviews, and Remote Work.
- ⚡ **Ultra-Fast LLM Inference via Groq API**: Powered by Groq API key (`gsk_...`) using `llama-3.3-70b-versatile` with automatic fallback to `llama3-8b-8192`.
- 🔍 **Grounded Answers & Citations**: High-precision vector similarity retrieval with FAISS. All answers display exact document source citations and text snippets.
- 🎨 **Executive UI/UX**: Streamlit dark mode UI with interactive Q&A chat, dashboard stats, preset question chips, document uploader, and active policy document browser.
- 📁 **Dynamic Document Upload**: HR managers or employees can upload new PDF, TXT, or MD policy files via sidebar to re-index the knowledge base on the fly.
- 🆓 **Zero Additional Embedding Cost**: Uses local `sentence-transformers/all-MiniLM-L6-v2` embeddings, eliminating third-party embedding API dependencies.

---

## 🛠️ Project Structure

```
company hr policy asst/
├── .env                       # API Key & Model Configuration
├── requirements.txt           # Python package dependencies
├── app.py                     # Streamlit Web Application
├── test_rag.py                # Automated RAG test script
├── run.bat                    # Easy Windows launcher
├── data/
│   └── hr_policies/           # Built-in HR Policy Knowledge Base
│       ├── leave_policy.md
│       ├── code_of_conduct.md
│       ├── benefits_and_wellness.md
│       ├── compensation_and_performance.md
│       └── remote_work_and_equipment.md
└── src/
    ├── config.py              # Configuration manager
    ├── document_loader.py     # Document loaders & chunking
    ├── vector_store.py        # FAISS vector store manager
    └── rag_engine.py          # LangChain Groq RAG chain
```

---

## ⚡ Quick Start

### 1. Install Dependencies
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run Automated RAG Test
```bash
python test_rag.py
```

### 3. Launch Streamlit Web UI
```bash
streamlit run app.py
```
*Or double-click `run.bat` on Windows.*

## 🌐 Deployment to Streamlit Community Cloud

Follow these quick steps to host your app live on **Streamlit Community Cloud**:

### 1. Create a GitHub Repository
1. Open [GitHub](https://github.com/) and log in with `janeshakaash02@gmail.com`.
2. Click **New Repository** and name it `company-hr-policy-asst`.
3. In your local terminal, run:
```bash
git init
git add .
git commit -m "Initial commit - HR Policy AI Assistant"
git branch -M main
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/company-hr-policy-asst.git
git push -u origin main
```

### 2. Deploy on Streamlit Community Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io) and log in with GitHub (`janeshakaash02@gmail.com`).
2. Click **New App** $\rightarrow$ Select your GitHub repository: `company-hr-policy-asst`.
3. Main file path: `app.py`.
4. Click **Advanced settings...** $\rightarrow$ Under **Secrets**, add:
```toml
GROQ_API_KEY = "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"
```
5. Click **Deploy!** Your app will be live with a shareable URL (e.g. `https://company-hr-policy-asst.streamlit.app`).

---

## 🔑 Groq API Key Configuration

The API key (`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`) is configured in `.env` and can also be entered live via the Web UI sidebar under **Advanced Settings**.
