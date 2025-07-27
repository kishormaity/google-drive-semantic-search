# evaluator.py

from langchain_anthropic import ChatAnthropic
import os
from dotenv import load_dotenv
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime

load_dotenv()

llm = ChatAnthropic(
    model="claude-3-haiku-20240307",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    temperature=0  # Ensure deterministic scoring
)

def evaluate_semantic_match(query: str, result: str) -> int:
    """
    Use an LLM to rate how well the result answers the query.
    Score: 0 = irrelevant, 1 = partial, 2 = full match
    """
    prompt = f"""
You are an evaluator checking how well a given document snippet answers a user query.

Query: {query}
Result: {result}

Rate the result:
- 2 = Fully answers the query
- 1 = Partially related or incomplete
- 0 = Irrelevant or incorrect

Respond ONLY with the number (0, 1, or 2).
"""

    try:
        response = llm.invoke(prompt)
        score = int(str(response.content).strip()[0])
        return score
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return -1  # Use -1 to indicate error

def evaluate_qa_response(query: str, answer: str, source_documents: List[str] = None) -> Dict:
    """
    Simple evaluation of QA response quality with essential metrics.
    Returns relevance, completeness, and overall score.
    """
    
    # 1. Relevance Score
    relevance_prompt = f"""
Evaluate how well this answer addresses the user's question.

Question: {query}
Answer: {answer}

Rate the relevance:
- 5 = Perfectly answers the question
- 4 = Very good answer with minor gaps
- 3 = Good answer but missing some details
- 2 = Partially relevant but incomplete
- 1 = Barely relevant
- 0 = Completely irrelevant

Respond ONLY with the number (0-5).
"""
    
    # 2. Completeness Score
    completeness_prompt = f"""
Evaluate how complete this answer is for the given question.

Question: {query}
Answer: {answer}

Rate the completeness:
- 5 = Complete answer with all necessary details
- 4 = Mostly complete with minor omissions
- 3 = Good coverage but missing some aspects
- 2 = Partial answer with significant gaps
- 1 = Very incomplete
- 0 = No useful information

Respond ONLY with the number (0-5).
"""
    
    try:
        # Get scores
        relevance_response = llm.invoke(relevance_prompt)
        relevance_score = int(str(relevance_response.content).strip()[0])
        
        completeness_response = llm.invoke(completeness_prompt)
        completeness_score = int(str(completeness_response.content).strip()[0])
        
        # Calculate overall score (simple average)
        overall_score = (relevance_score + completeness_score) / 2
        
        return {
            "relevance_score": relevance_score,
            "completeness_score": completeness_score,
            "overall_score": round(overall_score, 2),
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "answer_length": len(answer)
        }
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return {
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

def evaluate_retrieval_quality(query: str, retrieved_docs: List[str], expected_keywords: List[str] = None) -> Dict:
    """
    Simple evaluation of document retrieval quality.
    """
    
    if not retrieved_docs:
        return {
            "retrieval_score": 0,
            "documents_retrieved": 0,
            "issues": ["No documents retrieved"]
        }
    
    # Simple relevance evaluation
    relevance_scores = []
    for doc in retrieved_docs[:3]:  # Evaluate first 3 docs
        score = evaluate_semantic_match(query, doc[:500])  # Use first 500 chars
        if score >= 0:
            relevance_scores.append(score)
    
    avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
    retrieval_score = avg_relevance * 2.5  # Scale to 0-5

    return {
        "retrieval_score": round(retrieval_score, 2),
        "documents_retrieved": len(retrieved_docs),
        "issues": [] if retrieval_score >= 3 else ["Low relevance documents"]
    }

def run_evaluation_test_suite(vectorstore, test_queries: List[Tuple[str, str]] = None) -> Dict:
    """
    Run a simple evaluation test suite on the QA system.
    """
    
    if test_queries is None:
        test_queries = [
            ("What is the main topic?", "General knowledge question"),
            ("How does this work?", "Process explanation question"),
            ("What are the key points?", "Summary question")
        ]
    
    results = {
        "test_queries": len(test_queries),
        "average_scores": {},
        "detailed_results": []
    }
    
    all_scores = {
        "relevance": [],
        "completeness": [],
        "overall": []
    }
    
    for query, description in test_queries:
        try:
            # Get response from the system
            from main import query_llm
            response = query_llm(vectorstore, query, evaluate_response=False)
            
            # Extract just the answer part
            answer_lines = response.split('\n')
            answer = ""
            for line in answer_lines:
                if line.startswith('📌 **Answer**:'):
                    answer = line.replace('📌 **Answer**:', '').strip()
                    break
                elif not line.startswith('📎') and not line.startswith('📊'):
                    answer += line + " "
            
            # Evaluate the response
            evaluation = evaluate_qa_response(query, answer.strip())
            
            # Store results
            result_entry = {
                "query": query,
                "description": description,
                "answer": answer.strip(),
                "evaluation": evaluation
            }
            results["detailed_results"].append(result_entry)
            
            # Collect scores
            if "relevance_score" in evaluation:
                all_scores["relevance"].append(evaluation["relevance_score"])
                all_scores["completeness"].append(evaluation["completeness_score"])
                all_scores["overall"].append(evaluation["overall_score"])
                
        except Exception as e:
            print(f"❌ Test query failed: {query} - {e}")
    
    # Calculate averages
    for metric, scores in all_scores.items():
        if scores:
            results["average_scores"][metric] = round(sum(scores) / len(scores), 2)
    
    return results

def save_evaluation_results(results: Dict, filename: str = None) -> str:
    """
    Save evaluation results to a JSON file.
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"evaluation_results_{timestamp}.json"
    
    try:
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✅ Evaluation results saved to: {filename}")
        return filename
    except Exception as e:
        print(f"❌ Failed to save evaluation results: {e}")
        return None
