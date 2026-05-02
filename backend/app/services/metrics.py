"""
Advanced Evaluation Metrics — metrics.py

Objective metrics for AI quality:
  - Semantic Similarity (Cosine similarity via TF-IDF)
  - PII Detection (delegated to safety.py)
"""

import math
import re
from collections import Counter

def _get_tokens(text: str):
    """Simple tokenizer for pure-Python TF-IDF."""
    return re.findall(r"\w+", text.lower())

def calculate_semantic_similarity(text1: str, text2: str) -> float:
    """
    Calculate cosine similarity between two strings using TF-IDF.
    Pure Python implementation to avoid heavy dependencies while remaining
    more objective than Jaccard or Levenshtein.
    
    Returns: float (0.0 to 100.0)
    """
    if not text1 or not text2:
        return 0.0
        
    tokens1 = _get_tokens(text1)
    tokens2 = _get_tokens(text2)
    
    if not tokens1 or not tokens2:
        return 0.0
        
    all_tokens = set(tokens1).union(set(tokens2))
    
    # Calculate term frequency (simple bag of words for now)
    # For a two-document corpus, TF-IDF is basically just TF
    vec1 = Counter(tokens1)
    vec2 = Counter(tokens2)
    
    # Cosine Similarity
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum(vec1[x] * vec2[x] for x in intersection)
    
    sum1 = sum(vec1[x]**2 for x in vec1.keys())
    sum2 = sum(vec2[x]**2 for x in vec2.keys())
    denominator = math.sqrt(sum1) * math.sqrt(sum2)
    
    if not denominator:
        return 0.0
        
    score = (numerator / denominator) * 100.0
    return round(min(max(score, 0.0), 100.0), 1)

def detect_pii_leakage(text: str) -> bool:
    """
    Detect if any PII/PHI is present in the text.
    Delegates to the existing safety scanner.
    """
    from app.services.safety import scan_output
    scan = scan_output(text)
    return scan.get("pii_detected", False)
