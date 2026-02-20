import json
import time
import re
from typing import List, Dict
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_chroma import Chroma
from langchain_classic.chains import ConversationalRetrievalChain
from retrieval import get_local_retriever
from logger import logger
from sentence_transformers import CrossEncoder

def run_evaluation(llm: BaseChatModel, vectorstore: Chroma, eval_file: str = "rag_test_qa.json") -> Dict:
    """
    Runs evaluation using a hybrid approach:
    - Q&A generation via Gemini (API)
    - Scoring via Local Cross-Encoder (Local - Zero Quota used)
    """
    try:
        with open(eval_file, "r") as f:
            data = json.load(f)
            eval_data = data.get("rag_test_dataset", [])
    except Exception as e:
        logger.error(f"Failed to load evaluation data: {e}")
        return {"error": str(e)}

    # Initialize Local Judge (Uses the same model already in retrieval.py)
    logger.info("Initializing Local Judge model (ms-marco-MiniLM-L-6-v2)...")
    local_judge = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    results = []
    total_score = 0
    start_time = time.time()
    retriever = get_local_retriever(vectorstore)
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
    )

    for item in eval_data:
        # Pacing for Q&A generation
        time.sleep(2.0) 
        query = item.get("question")
        expected = item.get("expected_answer")
        
        logger.info(f"Generating answer for: {query}")
        try:
            # We ONLY hit the API for the answer generation
            response = qa_chain.invoke({"question": query, "chat_history": []})
            actual = response["answer"]
            source_docs = response.get("source_documents", [])
            sources = list(set([doc.metadata.get("source", "Unknown") for doc in source_docs]))
            
            # LOCAL SCORING (Zero API call)
            # Cross-encoder output is a similarity score. We normalize it to 0-10.
            similarity = local_judge.predict([(expected, actual)])
            # Access the first element if it's an array/numpy type
            sim_val = float(similarity[0]) if hasattr(similarity, "__len__") else float(similarity)
            
    # Map MS-Marco score to 0-10
            # Ensure float conversion for JSON compatibility
            score_val = float(round(max(0.0, min(10.0, (sim_val + 2.0) * 2.0)), 1))
            
            total_score += score_val
            results.append({
                "id": item.get("id"),
                "question": query,
                "expected": expected,
                "actual": actual,
                "sources": sources,
                "score": score_val
            })
        except Exception as e:
            logger.error(f"Failed to generate answer for '{query}': {e}")
            results.append({
                "id": item.get("id"),
                "question": query,
                "expected": expected,
                "actual": f"Error during execution: {str(e)}",
                "sources": [],
                "score": 0.0
            })

    total_time = float(time.time() - start_time)
    avg_score = float(total_score / len(eval_data)) if eval_data else 0.0

    return {
        "average_score": float(round(avg_score, 2)),
        "total_questions": int(len(eval_data)),
        "execution_time_sec": float(round(total_time, 2)),
        "results": results
    }
