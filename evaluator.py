# evaluator.py

from langchain_anthropic import ChatAnthropic
import os
from dotenv import load_dotenv
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
from main import query_llm  # Add missing import

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
        print(f"Evaluation failed: {e}")
        return -1  # Use -1 to indicate error

def evaluate_qa_response(query, response):
    """
    Evaluate the quality of a QA response.
    """
    try:
        # Parse the response to extract the answer
        lines = response.split('\n')
        answer = ""
        in_answer_section = False
        
        for line in lines:
            line = line.strip()
            if line.startswith('Answer:'):
                answer = line.replace('Answer:', '').strip()
                in_answer_section = True
            elif in_answer_section and line and not line.startswith('Sources Used:') and not line.startswith('Response Quality:'):
                answer += " " + line
            elif line.startswith('Sources Used:') or line.startswith('Response Quality:'):
                break
        
        if not answer:
            # Fallback: use the entire response
            answer = response
        
        # Calculate relevance score
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their', 'mine', 'yours', 'his', 'hers', 'ours', 'theirs'}
        
        query_words = query_words - stop_words
        answer_words = answer_words - stop_words
        
        if query_words:
            relevance_score = len(query_words.intersection(answer_words)) / len(query_words)
        else:
            relevance_score = 0
        
        # Calculate completeness score based on answer length and content
        min_length = 50
        max_length = 1000
        
        if len(answer) < min_length:
            completeness_score = len(answer) / min_length
        elif len(answer) > max_length:
            completeness_score = 1.0  # Full score for long answers
        else:
            completeness_score = 0.5 + (len(answer) - min_length) / (max_length - min_length) * 0.5
        
        # Calculate overall score
        overall_score = (relevance_score + completeness_score) / 2
        
        return {
            'relevance_score': round(relevance_score * 5, 2),
            'completeness_score': round(completeness_score * 5, 2),
            'overall_score': round(overall_score * 5, 2)
        }
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        return {
            'relevance_score': 0,
            'completeness_score': 0,
            'overall_score': 0
        }

def evaluate_retrieval_quality(vectorstore, query, retrieved_docs):
    """
    Evaluate the quality of document retrieval.
    """
    try:
        # Calculate average relevance score
        total_relevance = 0
        for doc in retrieved_docs:
            # Simple relevance calculation based on word overlap
            query_words = set(query.lower().split())
            doc_words = set(doc.page_content.lower().split())
            
            # Remove stop words
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            query_words = query_words - stop_words
            doc_words = doc_words - stop_words
            
            if query_words:
                relevance = len(query_words.intersection(doc_words)) / len(query_words)
            else:
                relevance = 0
            
            total_relevance += relevance
        
        avg_relevance = total_relevance / len(retrieved_docs) if retrieved_docs else 0
        
        # Scale to 0-5
        retrieval_score = avg_relevance * 2.5
        
        return {
            "retrieval_score": round(retrieval_score, 2),
            "documents_retrieved": len(retrieved_docs),
            "issues": [] if retrieval_score >= 3 else ["Low relevance documents"]
        }
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        return {
            "retrieval_score": 0,
            "documents_retrieved": 0,
            "issues": ["Evaluation error"]
        }

def run_comprehensive_evaluation(vectorstore, test_queries):
    """
    Run comprehensive evaluation on multiple test queries.
    """
    results = []
    
    for query in test_queries:
        try:
            # Get response
            response = query_llm(vectorstore, query, evaluate_response=False)
            
            # Parse response to extract answer
            lines = response.split('\n')
            answer = ""
            in_answer_section = False
            
            for line in lines:
                line = line.strip()
                if line.startswith('Answer:'):
                    answer = line.replace('Answer:', '').strip()
                    in_answer_section = True
                elif in_answer_section and line and not line.startswith('Sources Used:') and not line.startswith('Response Quality:'):
                    answer += " " + line
                elif line.startswith('Sources Used:') or line.startswith('Response Quality:'):
                    break
            
            if not answer:
                answer = response
            
            # Evaluate response
            evaluation = evaluate_qa_response(query, answer)
            
            results.append({
                'query': query,
                'answer': answer,
                'evaluation': evaluation
            })
            
        except Exception as e:
            print(f"Test query failed: {query} - {e}")
            results.append({
                'query': query,
                'answer': f"Error: {str(e)}",
                'evaluation': {'relevance_score': 0, 'completeness_score': 0, 'overall_score': 0}
            })
    
    return results

def save_evaluation_results(results, filename="evaluation_results.json"):
    """
    Save evaluation results to a JSON file.
    """
    try:
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Evaluation results saved to: {filename}")
    except Exception as e:
        print(f"Failed to save evaluation results: {e}")
