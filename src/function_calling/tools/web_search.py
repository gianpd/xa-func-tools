"""
/**
 * @file web_search_tool.py
 * @purpose Web search tool using SerpApi for Google search results with comprehensive error handling
 * 
 * @dependencies
 * - requests: For HTTP API calls to SerpApi
 * - os: For environment variable access
 * - json: For response parsing
 * - time: For rate limiting and delays
 * - typing: For type annotations
 *
 * @notes
 * - Requires SERPAPI_KEY environment variable
 * - Implements rate limiting to respect SerpApi limits
 * - Handles various search result types (organic, featured snippets, etc.)
 * - Provides structured output for LLM consumption
 * - Error handling for API failures and malformed responses
 */
"""

import os
import json
import time
import requests
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum

from dotenv import load_dotenv
load_dotenv()

class SearchResultType(Enum):
    """Types of search results returned by SerpApi"""
    ORGANIC = "organic"
    FEATURED_SNIPPET = "answer_box"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    NEWS = "news_results"
    IMAGES = "images_results"
    VIDEOS = "video_results"
    SHOPPING = "shopping_results"
    LOCAL = "local_results"

@dataclass
class SearchResult:
    """Structured search result data"""
    title: str
    link: str
    snippet: str
    source: str
    result_type: str
    position: Optional[int] = None
    date: Optional[str] = None
    thumbnail: Optional[str] = None

@dataclass
class SearchResponse:
    """Complete search response with metadata"""
    query: str
    results: List[SearchResult]
    total_results: int
    search_time: float
    featured_snippet: Optional[str] = None
    knowledge_graph: Optional[Dict[str, Any]] = None
    related_questions: Optional[List[str]] = None
    search_metadata: Optional[Dict[str, Any]] = None

class WebSearchTool:
    """
    Web search tool using SerpApi for comprehensive search functionality
    """
    
    def __init__(self, api_key: Optional[str] = None, rate_limit_delay: float = 1.0):
        """
        Initialize the web search tool
        
        Args:
            api_key: SerpApi API key (defaults to SERPAPI_KEY env var)
            rate_limit_delay: Delay between requests in seconds
        """
        self.api_key = api_key or os.getenv("SERPAPI_KEY")
        if not self.api_key:
            raise ValueError("SERPAPI_KEY environment variable or api_key parameter is required")
        
        self.base_url = "https://serpapi.com/search"
        self.rate_limit_delay = rate_limit_delay
        self.last_request_time = 0
        
        # Default search parameters
        self.default_params = {
            "engine": "google",
            "api_key": self.api_key,
            "num": 10,  # Number of results
            "safe": "active",  # Safe search
            "hl": "en",  # Language
            "gl": "us"  # Country
        }
    
    async def __aenter__(self):
        """
        Asynchronous context manager entry.
        No asynchronous setup is required for this tool, so it returns itself immediately.
        """
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Asynchronous context manager exit.
        No asynchronous cleanup is required for this tool.
        """
        pass
    
    def _respect_rate_limit(self):
        """Implement rate limiting to avoid hitting API limits"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _make_request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make HTTP request to SerpApi with error handling
        
        Args:
            params: Search parameters
            
        Returns:
            Raw API response as dictionary
            
        Raises:
            Exception: If request fails or returns error
        """
        self._respect_rate_limit()
        
        try:
            response = requests.get(
                self.base_url,
                params=params,
                timeout=30,
                headers={
                    "User-Agent": "WebSearchTool/1.0"
                }
            )
            
            response.raise_for_status()
            
            data = response.json()
            
            # Check for SerpApi errors
            if "error" in data:
                raise Exception(f"SerpApi error: {data['error']}")
            
            return data
            
        except requests.exceptions.Timeout:
            raise Exception("Search request timed out")
        except requests.exceptions.ConnectionError:
            raise Exception("Failed to connect to SerpApi")
        except requests.exceptions.HTTPError as e:
            raise Exception(f"HTTP error: {e.response.status_code} - {e.response.text}")
        except json.JSONDecodeError:
            raise Exception("Failed to parse API response as JSON")
        except Exception as e:
            raise Exception(f"Search request failed: {str(e)}")
    
    def _parse_organic_results(self, organic_results: List[Dict]) -> List[SearchResult]:
        """Parse organic search results"""
        results = []
        
        for i, result in enumerate(organic_results):
            search_result = SearchResult(
                title=result.get("title", ""),
                link=result.get("link", ""),
                snippet=result.get("snippet", ""),
                source=result.get("displayed_link", result.get("link", "")),
                result_type="organic",
                position=result.get("position", i + 1)
            )
            results.append(search_result)
        
        return results
    
    def _extract_featured_snippet(self, data: Dict) -> Optional[str]:
        """Extract featured snippet if available"""
        answer_box = data.get("answer_box")
        if answer_box:
            # Try different snippet fields
            snippet = (
                answer_box.get("snippet") or
                answer_box.get("answer") or
                answer_box.get("result") or
                ""
            )
            return snippet
        return None
    
    def _extract_knowledge_graph(self, data: Dict) -> Optional[Dict[str, Any]]:
        """Extract knowledge graph information"""
        kg = data.get("knowledge_graph")
        if kg:
            return {
                "title": kg.get("title"),
                "type": kg.get("type"),
                "description": kg.get("description"),
                "source": kg.get("source", {}).get("name"),
                "attributes": kg.get("attributes", {})
            }
        return None
    
    def _extract_related_questions(self, data: Dict) -> Optional[List[str]]:
        """Extract related questions (People Also Ask)"""
        related_questions = data.get("related_questions", [])
        if related_questions:
            return [q.get("question", "") for q in related_questions[:5]]
        return None
    
    def search(
        self,
        query: str,
        num_results: int = 10,
        result_types: Optional[List[str]] = None,
        location: Optional[str] = None,
        time_period: Optional[str] = None
    ) -> SearchResponse:
        """
        Perform web search using SerpApi
        
        Args:
            query: Search query string
            num_results: Number of results to return (1-100)
            result_types: Types of results to include (default: organic)
            location: Geographic location for search
            time_period: Time period filter (e.g., 'past_year', 'past_month')
            
        Returns:
            SearchResponse object with structured results
            
        Raises:
            Exception: If search fails
        """
        if not query.strip():
            raise ValueError("Search query cannot be empty")
        
        if num_results < 1 or num_results > 100:
            raise ValueError("num_results must be between 1 and 100")
        
        # Build search parameters
        params = self.default_params.copy()
        params.update({
            "q": query.strip(),
            "num": num_results
        })
        
        if location:
            params["location"] = location
        
        if time_period:
            params["tbs"] = f"qdr:{time_period}"
        
        search_start_time = time.time()
        
        # Make API request
        data = self._make_request(params)
        
        search_time = time.time() - search_start_time
        
        # Parse results
        organic_results = data.get("organic_results", [])
        parsed_results = self._parse_organic_results(organic_results)
        
        # Extract additional information
        featured_snippet = self._extract_featured_snippet(data)
        knowledge_graph = self._extract_knowledge_graph(data)
        related_questions = self._extract_related_questions(data)
        
        # Get search metadata
        search_info = data.get("search_information", {})
        total_results = search_info.get("total_results", len(parsed_results))
        
        if isinstance(total_results, str):
            # Parse total results if it's a string like "About 1,234,567 results"
            try:
                total_results = int(''.join(filter(str.isdigit, total_results)))
            except ValueError:
                total_results = len(parsed_results)
        
        return SearchResponse(
            query=query,
            results=parsed_results,
            total_results=total_results,
            search_time=search_time,
            featured_snippet=featured_snippet,
            knowledge_graph=knowledge_graph,
            related_questions=related_questions,
            search_metadata=search_info
        )
    
    def search_news(
        self,
        query: str,
        num_results: int = 10,
        time_period: Optional[str] = None
    ) -> SearchResponse:
        """
        Search for news articles
        
        Args:
            query: Search query
            num_results: Number of news results
            time_period: Time filter (e.g., 'past_day', 'past_week')
            
        Returns:
            SearchResponse with news results
        """
        params = self.default_params.copy()
        params.update({
            "q": query.strip(),
            "tbm": "nws",  # News search
            "num": num_results
        })
        
        if time_period:
            params["tbs"] = f"qdr:{time_period}"
        
        search_start_time = time.time()
        data = self._make_request(params)
        search_time = time.time() - search_start_time
        
        # Parse news results
        news_results = data.get("news_results", [])
        parsed_results = []
        
        for i, result in enumerate(news_results):
            search_result = SearchResult(
                title=result.get("title", ""),
                link=result.get("link", ""),
                snippet=result.get("snippet", ""),
                source=result.get("source", ""),
                result_type="news",
                position=i + 1,
                date=result.get("date"),
                thumbnail=result.get("thumbnail")
            )
            parsed_results.append(search_result)
        
        return SearchResponse(
            query=query,
            results=parsed_results,
            total_results=len(parsed_results),
            search_time=search_time
        )

# Tool function for use with the FunctionCalling framework
def web_search(
    query: str,
    num_results: int = 5,
    search_type: str = "web",
    location: Optional[str] = None,
    time_period: Optional[str] = None
) -> str:
    """
    Search the web using SerpApi and return formatted results
    
    Args:
        query: The search query string
        num_results: Number of results to return (1-10, default: 5)
        search_type: Type of search - "web" or "news" (default: "web")
        location: Geographic location for search (optional)
        time_period: Time filter - "past_day", "past_week", "past_month", "past_year" (optional)
    
    Returns:
        Formatted search results as a string
        
    Examples:
        web_search("artificial intelligence trends 2024")
        web_search("climate change", num_results=3, time_period="past_month")
        web_search("restaurants near me", location="New York, NY")
        web_search("breaking news AI", search_type="news", num_results=5)
    """
    try:
        # Initialize search tool
        search_tool = WebSearchTool()
        
        # Validate inputs
        num_results = max(1, min(num_results, 10))  # Clamp between 1-10
        
        # Perform search based on type
        if search_type.lower() == "news":
            response = search_tool.search_news(
                query=query,
                num_results=num_results,
                time_period=time_period
            )
        else:
            response = search_tool.search(
                query=query,
                num_results=num_results,
                location=location,
                time_period=time_period
            )
        
        # Format results for LLM consumption
        formatted_results = f"🔍 **Search Results for:** {query}\n"
        formatted_results += f"📊 **Found:** {response.total_results:,} total results\n"
        formatted_results += f"⏱️ **Search time:** {response.search_time:.2f} seconds\n\n"
        
        # Add featured snippet if available
        if response.featured_snippet:
            formatted_results += f"💡 **Featured Answer:**\n{response.featured_snippet}\n\n"
        
        # Add knowledge graph if available
        if response.knowledge_graph:
            kg = response.knowledge_graph
            formatted_results += f"📚 **Knowledge Graph:**\n"
            formatted_results += f"**{kg.get('title', 'N/A')}** ({kg.get('type', 'N/A')})\n"
            if kg.get('description'):
                formatted_results += f"{kg['description']}\n"
            formatted_results += f"*Source: {kg.get('source', 'N/A')}*\n\n"
        
        # Add search results
        if response.results:
            formatted_results += f"📋 **Top {len(response.results)} Results:**\n\n"
            
            for i, result in enumerate(response.results, 1):
                formatted_results += f"**{i}. {result.title}**\n"
                formatted_results += f"🔗 {result.link}\n"
                if result.snippet:
                    formatted_results += f"📝 {result.snippet}\n"
                if result.date:
                    formatted_results += f"📅 {result.date}\n"
                formatted_results += f"🌐 Source: {result.source}\n\n"
        else:
            formatted_results += "❌ No results found for this query.\n"
        
        # Add related questions if available
        if response.related_questions:
            formatted_results += f"❓ **Related Questions:**\n"
            for question in response.related_questions:
                formatted_results += f"• {question}\n"
            formatted_results += "\n"
        
        return formatted_results.strip()
        
    except ValueError as e:
        return f"❌ **Search Error:** Invalid input - {str(e)}"
    except Exception as e:
        return f"❌ **Search Error:** {str(e)}"

# Example usage and testing
if __name__ == "__main__":
    # Test the web search tool
    try:
        # Test basic search
        result = web_search("Python programming best practices", num_results=3)
        print("=== Basic Search Test ===")
        print(result)
        print("\n" + "="*50 + "\n")
        
        # Test news search
        news_result = web_search("AI developments", search_type="news", num_results=3, time_period="past_week")
        print("=== News Search Test ===")
        print(news_result)
        
    except Exception as e:
        print(f"Test failed: {e}")
        
    # Test with FunctionCalling framework
    print("\n=== Function Schema ===")
    from inspect import signature
    print(f"Function: {web_search.__name__}")
    print(f"Signature: {signature(web_search)}")
    print(f"Docstring: {web_search.__doc__}")