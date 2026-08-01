import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.rag_engine import HRPolicyRAGEngine
from src.config import get_groq_api_key

def test_rag_pipeline():
    print("==================================================")
    print("Testing HR Policy RAG Pipeline...")
    print("==================================================")
    
    api_key = get_groq_api_key()
    print(f"API Key present: {'Yes' if api_key else 'No'}")
    
    engine = HRPolicyRAGEngine(api_key=api_key)
    
    test_queries = [
        "How many annual PTO days do full-time employees receive?",
        "What is the policy for parental leave?",
        "What is the internet reimbursement amount for remote workers?",
        "What is the POSH (Prevention of Sexual Harassment) policy and how do I file a complaint with the ICC?",
        "Can I claim expenses for buying space shuttles?" # Out-of-bounds question to test fallback
    ]
    
    for q in test_queries:
        print(f"\n[Query]: {q}")
        res = engine.query(q)
        print(f"[Answer]: {res['answer']}\n")
        print(f"[Sources]: {[s['file'] for s in res['sources']]}")
        print("-" * 50)
        
    print("\nRAG Pipeline Test Completed Successfully!")

if __name__ == "__main__":
    test_rag_pipeline()
