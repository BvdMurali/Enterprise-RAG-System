import sys
from pathlib import Path
import time
import logging

# Add project root directory to path to resolve imports
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

from backend.config import get_settings
from backend.models.llm import LLMService
from backend.services.embeddings import EmbeddingService
from backend.vectorstore.chroma_store import ChromaStoreService
from backend.services.retriever import RetrieverService
from backend.services.rag_pipeline import RAGPipelineService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("evaluate_rag")

# Faithfulness evaluator instructions (checks if answer is fully backed by context)
FAITHFULNESS_EVAL_PROMPT = """You are an independent RAG evaluation judge. Your task is to evaluate the FAITHFULNESS of a generated answer compared to the provided document context.
Faithfulness measures if all facts in the generated answer are strictly supported by the context.

Retrieved Context:
{context}

Generated Answer:
{answer}

Rules:
1. Rate the Faithfulness on a scale of 0.0 to 1.0.
   - 1.0: Everything in the answer is fully supported by the context.
   - 0.5: Part of the answer is supported, but there are ungrounded assumptions or facts not in the context.
   - 0.0: The answer has no support or contradicts the context.
2. Return ONLY a single float rating between 0.0 and 1.0. Do NOT explain your reasoning.

Rating:"""

# Relevance evaluator instructions (checks if answer responds to user query)
RELEVANCE_EVAL_PROMPT = """You are an independent RAG evaluation judge. Your task is to evaluate the RELEVANCE of the generated answer compared to the original user question.
Answer Relevance measures if the answer addresses the question directly without irrelevant boilerplate.

User Question:
{question}

Generated Answer:
{answer}

Rules:
1. Rate the Answer Relevance on a scale of 0.0 to 1.0.
   - 1.0: The answer directly and fully addresses the question.
   - 0.5: The answer is partially relevant or too generic.
   - 0.0: The answer completely misses the point of the question.
2. Return ONLY a single float rating between 0.0 and 1.0. Do NOT explain your reasoning.

Rating:"""


def run_custom_evaluation():
    logger.info("Initializing services for evaluation...")
    settings = get_settings()
    
    try:
        emb_service = EmbeddingService()
        chroma_store = ChromaStoreService(embeddings=emb_service.embeddings)
        retriever = RetrieverService(chroma_store=chroma_store)
        llm_service = LLMService()
        pipeline = RAGPipelineService(retriever_service=retriever, llm_service=llm_service)
    except Exception as e:
        logger.error(f"Failed to initialize evaluation services: {e}. Make sure GOOGLE_API_KEY is configured.")
        return

    # Check if database contains vectors
    collection_data = retriever.chroma_store.vector_store.get()
    contents = collection_data.get("documents", [])
    if not contents:
        logger.warning("No documents found in database. Ingest a document before running evaluation.")
        return

    test_queries = [
        "What is the main topic of the document?",
        "Summarize the key requirements or policies.",
        "List any structural details or metrics."
    ]

    results = []
    
    for q in test_queries:
        logger.info(f"Evaluating query: '{q}'")
        
        start_time = time.time()
        res = pipeline.ask(q)
        latency = time.time() - start_time
        
        answer = res.get("answer", "")
        sources = res.get("sources", [])
        
        # Format context string
        context_str = "\n\n".join([f"- {s['content']}" for s in sources]) if sources else "No context retrieved."
        
        # Call Gemini to score metrics
        llm = llm_service.get_llm()
        
        faith_score = 0.0
        try:
            faith_response = llm.invoke(FAITHFULNESS_EVAL_PROMPT.format(context=context_str, answer=answer))
            faith_score = float(faith_response.content.strip())
        except Exception as e:
            logger.error(f"Faithfulness eval failed: {e}")
            
        relevance_score = 0.0
        try:
            rel_response = llm.invoke(RELEVANCE_EVAL_PROMPT.format(question=q, answer=answer))
            relevance_score = float(rel_response.content.strip())
        except Exception as e:
            logger.error(f"Relevance eval failed: {e}")

        result_item = {
            "question": q,
            "latency_seconds": round(latency, 2),
            "faithfulness": faith_score,
            "answer_relevance": relevance_score,
            "sources_count": len(sources)
        }
        results.append(result_item)
        logger.info(f"Evaluation metrics: Latency={latency:.2f}s, Faithfulness={faith_score}, Relevance={relevance_score}")

    # Generate execution summary
    print("\n" + "="*50)
    print("           RAG EVALUATION PIPELINE REPORT")
    print("="*50)
    print(f"Total Queries Evaluated: {len(results)}")
    avg_latency = sum(r["latency_seconds"] for r in results) / len(results)
    avg_faith = sum(r["faithfulness"] for r in results) / len(results)
    avg_relevance = sum(r["answer_relevance"] for r in results) / len(results)
    
    print(f"Average Latency:      {avg_latency:.2f} seconds")
    print(f"Average Faithfulness: {avg_faith:.2f} (Target: >0.85)")
    print(f"Average Relevance:    {avg_relevance:.2f} (Target: >0.80)")
    print("-"*50)
    
    for idx, r in enumerate(results, 1):
        print(f"Q{idx}: '{r['question']}'")
        print(f"   - Latency:      {r['latency_seconds']}s")
        print(f"   - Faithfulness: {r['faithfulness']}")
        print(f"   - Relevance:    {r['answer_relevance']}")
    print("="*50 + "\n")


if __name__ == "__main__":
    run_custom_evaluation()
