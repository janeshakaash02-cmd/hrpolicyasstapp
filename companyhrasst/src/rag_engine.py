import os
from typing import Dict, Any, List
from pathlib import Path

# Groq LLM import
try:
    from langchain_groq import ChatGroq
except ImportError:
    raise ImportError("Please install langchain-groq: pip install langchain-groq")

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from src.config import get_groq_api_key, DEFAULT_MODEL, FALLBACK_MODEL, TOP_K_RESULTS
from src.vector_store import VectorStoreManager

# Strict System Prompt for HR Policy Assistant
HR_SYSTEM_PROMPT = """You are the official Acme Corp HR Policy Assistant. 
Your primary job is to provide clear, accurate, and professional answers to employees regarding company HR policies, benefits, leave, code of conduct, and workplace procedures based ONLY on the retrieved policy context provided below.

Guidelines:
1. Base your answer STRICTLY on the provided Context documents. Do NOT make up information or speculate outside the HR policies.
2. If the answer cannot be found in the provided context, state clearly and politely: 
   "I could not find specific information regarding this topic in our current HR Policy documents. Please reach out directly to the HR Department at hr@acmecorp.com for assistance."
3. Keep your tone professional, empathetic, clear, and helpful.
4. Use bullet points or numbered lists where appropriate to make information easy to digest.
5. Highlight important numbers, amounts, deadlines, or rules in bold.

Context:
{context}

Question:
{input}

Answer:"""

class HRPolicyRAGEngine:
    """Core RAG Engine integrating Groq LLM with FAISS Vector Store."""
    
    def __init__(self, api_key: str = None, model_name: str = DEFAULT_MODEL):
        self.api_key = get_groq_api_key(api_key)
        self.model_name = model_name
        self.vector_manager = VectorStoreManager()
        self.llm = None
        self._init_llm()

    def _init_llm(self):
        """Initialize ChatGroq instance with fallback handling."""
        if not self.api_key:
            raise ValueError("Groq API Key is missing. Please provide a valid API key.")
            
        try:
            self.llm = ChatGroq(
                groq_api_key=self.api_key,
                model_name=self.model_name,
                temperature=0.2,
                max_tokens=1024
            )
        except Exception as e:
            print(f"Error initializing {self.model_name}: {e}. Retrying with fallback model {FALLBACK_MODEL}...")
            self.llm = ChatGroq(
                groq_api_key=self.api_key,
                model_name=FALLBACK_MODEL,
                temperature=0.2,
                max_tokens=1024
            )

    def query(self, user_question: str) -> Dict[str, Any]:
        """Query the RAG pipeline and return answer along with source document metadata."""
        user_question = (user_question or "").strip()
        if not user_question:
            return {
                "answer": "Please ask a valid question regarding Acme Corp HR Policies.",
                "sources": [],
                "error": None
            }

        try:
            # 1. Retrieve relevant document chunks (compatible with all LangChain versions)
            vector_store = self.vector_manager.get_or_create_vector_store()
            retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K_RESULTS})
            
            if hasattr(retriever, "invoke"):
                relevant_docs = retriever.invoke(user_question)
            elif hasattr(retriever, "get_relevant_documents"):
                relevant_docs = retriever.get_relevant_documents(user_question)
            else:
                relevant_docs = vector_store.similarity_search(user_question, k=TOP_K_RESULTS)

            if not relevant_docs:
                return {
                    "answer": "No relevant HR policy documents found.",
                    "sources": [],
                    "error": None
                }

            # 2. Extract context & source metadata
            context_str = "\n\n---\n\n".join([doc.page_content for doc in relevant_docs])
            sources = []
            seen_sources = set()
            for doc in relevant_docs:
                src_name = doc.metadata.get("source_filename") or doc.metadata.get("source", "HR Policy Document")
                src_basename = Path(src_name).name
                if src_basename not in seen_sources:
                    seen_sources.add(src_basename)
                    sources.append({
                        "file": src_basename,
                        "snippet": doc.page_content[:250] + "..."
                    })

            # 3. Generate response using ChatGroq & Prompt
            prompt_template = ChatPromptTemplate.from_template(HR_SYSTEM_PROMPT)
            messages = prompt_template.format_messages(
                context=context_str,
                input=user_question
            )

            try:
                response = self.llm.invoke(messages)
                answer_text = response.content
            except Exception as llm_err:
                try:
                    if self.model_name != FALLBACK_MODEL:
                        fallback_llm = ChatGroq(
                            groq_api_key=self.api_key,
                            model_name=FALLBACK_MODEL,
                            temperature=0.2,
                            max_tokens=1024
                        )
                        response = fallback_llm.invoke(messages)
                        answer_text = response.content
                    else:
                        raise llm_err
                except Exception:
                    # Smart synthesis fallback when API call fails
                    extracted = []
                    for doc in relevant_docs:
                        for line in doc.page_content.split("\n"):
                            line_str = line.strip()
                            if line_str.startswith("-") or line_str.startswith("*") or line_str.startswith("Answer:") or ":" in line_str:
                                if len(line_str) > 20 and not "DECLARATION" in line_str:
                                    extracted.append(line_str)
                    if not extracted:
                        extracted = [relevant_docs[0].page_content[:300]]
                    bullets = "\n".join([f"- {b.lstrip('-*• ')}" for b in extracted[:5]])
                    answer_text = (
                        f"### 📌 Executive Overview\nBased on official Acme Corp HR policies regarding **'{user_question}'**:\n\n"
                        f"### 📋 Detailed Policy Provisions\n{bullets}\n\n"
                        f"### 💡 Action Steps & Guidelines\n- Submit requests via the Acme Corp HR Portal.\n- Contact HR Support at `hr@acmecorp.com`."
                    )

            return {
                "answer": answer_text,
                "sources": sources,
                "error": None
            }

        except Exception as e:
            return {
                "answer": f"An error occurred while processing your query: {str(e)}",
                "sources": [],
                "error": str(e)
            }

    def reindex(self, data_dir: Path = None):
        """Force re-building of vector database."""
        if data_dir:
            self.vector_manager.build_from_directory(data_dir)
        else:
            self.vector_manager.build_from_directory()
