"""
 * @file web_search_tool.py
 * @purpose Web search tool using SerpApi for Google search results with comprehensive error handling
 * 
 * @dependencies
 * - aiohttp: For async HTTP API calls to SerpApi
 * - asyncio: For async operations and sleep
 * - os: For environment variable access
 * - json: For response parsing
 * - time: For timing measurements
 * - typing: For type annotations
 *
 * @notes
 * - Requires SERPAPI_KEY environment variable
 * - Implements rate limiting to respect SerpApi limits
 * - Handles various search result types (organic, featured snippets, etc.)
 * - Provides structured output for LLM consumption
 * - Error handling for API failures and malformed responses
 * - Fully asynchronous implementation using aiohttp
"""

import os
import json
import time
import asyncio
import aiohttp
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
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Default search parameters
        self.default_params = {
            "engine": "google",
            "api_key": self.api_key,
            "num": 10,
            "safe": "active",
            "hl": "en",
            "gl": "us"
        }
    
    async def __aenter__(self):
        """
        Asynchronous context manager entry.
        Creates an aiohttp session for making requests.
        """
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={"User-Agent": "WebSearchTool/1.0"}
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Asynchronous context manager exit.
        Closes the aiohttp session.
        """
        if self.session:
            await self.session.close()
    
    async def _respect_rate_limit(self):
        """Implement async rate limiting to avoid hitting API limits"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last
            await asyncio.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    async def _make_request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make async HTTP request to SerpApi with error handling
        
        Args:
            params: Search parameters
            
        Returns:
            Raw API response as dictionary
            
        Raises:
            Exception: If request fails or returns error
        """
        await self._respect_rate_limit()
        
        if not self.session:
            raise RuntimeError("WebSearchTool must be used as an async context manager")
        
        try:
            async with self.session.get(
                self.base_url,
                params=params
            ) as response:
                response.raise_for_status()
                data = await response.json()
                
                # Check for SerpApi errors
                if "error" in data:
                    raise Exception(f"SerpApi error: {data['error']}")
                
                return data
                
        except asyncio.TimeoutError:
            raise Exception("Search request timed out")
        except aiohttp.ClientConnectionError:
            raise Exception("Failed to connect to SerpApi")
        except aiohttp.ClientResponseError as e:
            raise Exception(f"HTTP error: {e.status} - {e.message}")
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
    
    def _parse_news_results(self, news_results: List[Dict]) -> List[SearchResult]:
        """Parse news search results"""
        results = []
        
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
            results.append(search_result)
        
        return results
    
    def _extract_featured_snippet(self, data: Dict) -> Optional[str]:
        """Extract featured snippet if available"""
        answer_box = data.get("answer_box")
        if answer_box:
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
    
    async def search(
        self,
        query: str,
        num_results: int = 10,
        parse_result_for_llm=True,
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
            parse_result_for_llm: Boolean deciding if or not parsing result
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
        data = await self._make_request(params)
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
            try:
                total_results = int(''.join(filter(str.isdigit, total_results)))
            except ValueError:
                total_results = len(parsed_results)
        
        result = SearchResponse(
            query=query,
            results=parsed_results,
            total_results=total_results,
            search_time=search_time,
            featured_snippet=featured_snippet,
            knowledge_graph=knowledge_graph,
            related_questions=related_questions,
            search_metadata=search_info
        )

        return self.format_results_for_llm(response=result) if parse_result_for_llm else result
    
    async def search_news(
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
            "tbm": "nws",
            "num": num_results
        })
        
        if time_period:
            params["tbs"] = f"qdr:{time_period}"
        
        search_start_time = time.time()
        data = await self._make_request(params)
        search_time = time.time() - search_start_time
        
        # Parse news results
        news_results = data.get("news_results", [])
        parsed_results = self._parse_news_results(news_results)
        
        search_info = data.get("search_information", {})
        total_results = len(parsed_results)
        
        return SearchResponse(
            query=query,
            results=parsed_results,
            total_results=total_results,
            search_time=search_time,
            search_metadata=search_info
        )
    
    async def search_images(
        self,
        query: str,
        num_results: int = 10,
        safe_search: bool = True
    ) -> SearchResponse:
        """
        Search for images
        
        Args:
            query: Search query
            num_results: Number of image results
            safe_search: Enable safe search filtering
            
        Returns:
            SearchResponse with image results
        """
        params = self.default_params.copy()
        params.update({
            "q": query.strip(),
            "tbm": "isch",
            "num": num_results,
            "safe": "active" if safe_search else "off"
        })
        
        search_start_time = time.time()
        data = await self._make_request(params)
        search_time = time.time() - search_start_time
        
        # Parse image results
        images = data.get("images_results", [])
        results = []
        
        for i, img in enumerate(images):
            search_result = SearchResult(
                title=img.get("title", ""),
                link=img.get("original", ""),
                snippet=img.get("snippet", ""),
                source=img.get("source", ""),
                result_type="image",
                position=i + 1,
                thumbnail=img.get("thumbnail")
            )
            results.append(search_result)
        
        return SearchResponse(
            query=query,
            results=results,
            total_results=len(results),
            search_time=search_time,
            search_metadata=data.get("search_information", {})
        )
    
    def format_results_for_llm(self, response: SearchResponse, max_results: int = 5) -> str:
        """
        Format search results for LLM consumption
        
        Args:
            response: SearchResponse object
            max_results: Maximum number of results to include
            
        Returns:
            Formatted string suitable for LLM context
        """
        output = [f"Search Query: {response.query}"]
        output.append(f"Total Results: {response.total_results:,}")
        output.append(f"Search Time: {response.search_time:.2f}s\n")
        
        if response.featured_snippet:
            output.append(f"Featured Snippet:\n{response.featured_snippet}\n")
        
        if response.knowledge_graph:
            kg = response.knowledge_graph
            output.append(f"Knowledge Graph:")
            output.append(f"  Title: {kg.get('title', 'N/A')}")
            output.append(f"  Type: {kg.get('type', 'N/A')}")
            output.append(f"  Description: {kg.get('description', 'N/A')}\n")
        
        output.append("Search Results:")
        for i, result in enumerate(response.results[:max_results], 1):
            output.append(f"\n{i}. {result.title}")
            output.append(f"   URL: {result.link}")
            output.append(f"   Snippet: {result.snippet}")
            if result.date:
                output.append(f"   Date: {result.date}")
        
        if response.related_questions:
            output.append("\nRelated Questions:")
            for q in response.related_questions:
                output.append(f"  - {q}")
        
        return "\n".join(output)


# Example usage
async def main():
    """Example usage of WebSearchTool"""
    async with WebSearchTool() as search_tool:
        # Standard web search
        print("="*60)
        print("STANDARD WEB SEARCH")
        print("="*60)
        response = await search_tool.search(
            query="Python async programming best practices",
            num_results=5,
            parse_result_for_llm=True
        )
        print(response)
        
        print("\n" + "="*60)
        print("NEWS SEARCH")
        print("="*60)
        # News search
        news_response = await search_tool.search_news(
            query="artificial intelligence breakthroughs",
            num_results=5,
            time_period="past_week"
        )
        print(search_tool.format_results_for_llm(news_response))


if __name__ == "__main__":
    asyncio.run(main())