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
    Comprehensive evaluation of QA response quality.
    Returns detailed metrics for the response.
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
    
    # 3. Source Relevance (if source documents provided)
    source_score = None
    if source_documents:
        source_text = "\n".join(source_documents[:3])  # Use first 3 sources
        source_prompt = f"""
Evaluate how relevant the source documents are to the question.

Question: {query}
Source Documents: {source_text}

Rate the source relevance:
- 5 = Sources perfectly match the question
- 4 = Sources are very relevant
- 3 = Sources are somewhat relevant
- 2 = Sources are barely relevant
- 1 = Sources are not relevant
- 0 = No sources provided

Respond ONLY with the number (0-5).
"""
    
    try:
        # Get scores
        relevance_response = llm.invoke(relevance_prompt)
        relevance_score = int(str(relevance_response.content).strip()[0])
        
        completeness_response = llm.invoke(completeness_prompt)
        completeness_score = int(str(completeness_response.content).strip()[0])
        
        if source_documents:
            source_response = llm.invoke(source_prompt)
            source_score = int(str(source_response.content).strip()[0])
        
        # Calculate overall score
        scores = [relevance_score, completeness_score]
        if source_score is not None:
            scores.append(source_score)
        
        overall_score = sum(scores) / len(scores)
        
        return {
            "relevance_score": relevance_score,
            "completeness_score": completeness_score,
            "source_relevance_score": source_score,
            "overall_score": round(overall_score, 2),
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "answer_length": len(answer)
        }
        
    except Exception as e:
        print(f"❌ Comprehensive evaluation failed: {e}")
        return {
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

def evaluate_retrieval_quality(query: str, retrieved_docs: List[str], expected_keywords: List[str] = None) -> Dict:
    """
    Evaluate the quality of document retrieval for a query.
    """
    
    if not retrieved_docs:
        return {
            "retrieval_score": 0,
            "coverage_score": 0,
            "diversity_score": 0,
            "error": "No documents retrieved"
        }
    
    # Join retrieved documents
    docs_text = "\n---\n".join(retrieved_docs[:5])  # Limit to first 5 docs
    
    # 1. Retrieval Relevance
    retrieval_prompt = f"""
Evaluate how relevant the retrieved documents are to the query.

Query: {query}
Retrieved Documents: {docs_text}

Rate the retrieval relevance:
- 5 = All documents highly relevant
- 4 = Most documents relevant
- 3 = Some documents relevant
- 2 = Few documents relevant
- 1 = Most documents irrelevant
- 0 = All documents irrelevant

Respond ONLY with the number (0-5).
"""
    
    # 2. Coverage Score
    coverage_prompt = f"""
Evaluate how well the retrieved documents cover the information needed for the query.

Query: {query}
Retrieved Documents: {docs_text}

Rate the coverage:
- 5 = Complete coverage of all aspects
- 4 = Good coverage with minor gaps
- 3 = Adequate coverage
- 2 = Poor coverage with significant gaps
- 1 = Very poor coverage
- 0 = No relevant coverage

Respond ONLY with the number (0-5).
"""
    
    try:
        retrieval_response = llm.invoke(retrieval_prompt)
        retrieval_score = int(str(retrieval_response.content).strip()[0])
        
        coverage_response = llm.invoke(coverage_prompt)
        coverage_score = int(str(coverage_response.content).strip()[0])
        
        # Calculate diversity (based on document count and content variety)
        diversity_score = min(5, len(retrieved_docs) * 1.5)  # Simple heuristic
        
        return {
            "retrieval_score": retrieval_score,
            "coverage_score": coverage_score,
            "diversity_score": round(diversity_score, 2),
            "num_documents": len(retrieved_docs),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"❌ Retrieval evaluation failed: {e}")
        return {
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

def run_evaluation_test_suite(vectorstore, test_queries: List[Tuple[str, str]] = None) -> Dict:
    """
    Run a comprehensive evaluation test suite on your QA system.
    
    Args:
        vectorstore: Your FAISS vectorstore
        test_queries: List of (query, expected_answer) tuples for testing
    """
    
    if test_queries is None:
        # Default test queries for document QA
        test_queries = [
            ("What is the main topic of the documents?", "General document content"),
            ("Can you summarize the key points?", "Document summary"),
            ("What are the important dates mentioned?", "Date information"),
            ("Who are the main people mentioned?", "Person names"),
            ("What are the main conclusions?", "Conclusions")
        ]
    
    results = {
        "test_suite_results": [],
        "overall_stats": {},
        "timestamp": datetime.now().isoformat()
    }
    
    total_scores = []
    
    for query, expected in test_queries:
        try:
            # Get response from your system
            from main import query_llm
            response = query_llm(vectorstore, query)
            
            # Extract answer from response (remove sources)
            if "📌 **Answer**:\n" in response:
                answer = response.split("📌 **Answer**:\n")[1].split("\n\n📎 **Sources**:")[0]
            else:
                answer = response
            
            # Evaluate the response
            evaluation = evaluate_qa_response(query, answer)
            
            test_result = {
                "query": query,
                "expected": expected,
                "actual_answer": answer,
                "evaluation": evaluation,
                "response_length": len(answer)
            }
            
            results["test_suite_results"].append(test_result)
            
            if "overall_score" in evaluation:
                total_scores.append(evaluation["overall_score"])
                
        except Exception as e:
            print(f"❌ Test failed for query '{query}': {e}")
            results["test_suite_results"].append({
                "query": query,
                "error": str(e)
            })
    
    # Calculate overall statistics
    if total_scores:
        results["overall_stats"] = {
            "average_score": round(sum(total_scores) / len(total_scores), 2),
            "max_score": max(total_scores),
            "min_score": min(total_scores),
            "total_tests": len(total_scores),
            "success_rate": round(len(total_scores) / len(test_queries) * 100, 2)
        }
    
    return results

def save_evaluation_results(results: Dict, filename: str = None) -> str:
    """
    Save evaluation results to a JSON file.
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"evaluation_results_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Evaluation results saved to: {filename}")
    return filename
