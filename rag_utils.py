import os
import logging
from typing import List, Dict, Any

logger = logging.getLogger("rag_utils")

def build_rag_prompt(question: str, context: str) -> str:
    """
    Helper to build a prompt with context.
    """
    return f"""
    Context:
    {context}
    
    Question:
    {question}
    """

def search_web(query: str, api_key: str = None, num: int = 3) -> str:
    """
    Search the web using SerpAPI (if key provided).
    Returns a string of concatenated snippets.
    """
    if not api_key:
        # Check env if not provided
        api_key = os.getenv("SERPAPI_API_KEY")
        
    if not api_key:
        logger.warning("No SERPAPI_API_KEY provided. Web search disabled.")
        return ""
        
    try:
        from serpapi import GoogleSearch
        params = {"q": query, "engine": "google", "api_key": api_key, "num": num}
        search = GoogleSearch(params)
        res = search.get_dict()
        snippets = []
        for r in res.get("organic_results", [])[:num]:
            title = r.get("title","")
            snippet = r.get("snippet","")
            url = r.get("link","")
            snippets.append(f"Title: {title}\nSnippet: {snippet}\nSource: {url}")
        
        return "\n\n".join(snippets)
    except ImportError:
        logger.error("google-search-results not installed. Run `pip install google-search-results`.")
        return ""
    except Exception as e:
        logger.error(f"Web search failed: {e}")
        return ""
