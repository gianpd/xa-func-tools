#!/usr/bin/env python3
"""
Context-Aware Web Research Pipeline - CORRECTED

Combines web search, URL quality analysis, LLM-based relevance ranking,
and intelligent content scraping into a unified pipeline for high-quality
information retrieval.

Key fixes:
1. Updated to use SearchResult attributes correctly (url -> link)
2. Added proper rank/confidence handling from SearchResult
3. Fixed engine attribute access
4. Corrected async/await patterns for LLM callable
"""

import asyncio
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any, Callable
from datetime import datetime
from urllib.parse import urlparse

# Import tools from the codebase
from src.function_calling.tools.url_context import URLContextTool
from src.function_calling.tools.web_search import WebSearchTool, SearchResponse, SearchResult
from src.function_calling.tools.web_scraper import WebScraper, ScrapingConfig


logger = logging.getLogger(__name__)


@dataclass
class URLQualityScore:
    """Quality assessment for a URL."""
    url: str
    title: str
    snippet: str
    domain: str
    quality_score: float  # 0.0 - 1.0
    security_score: int  # 0 - 100
    security_level: str  # LOW, MEDIUM, HIGH
    is_secure: bool
    issues: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ResearchResult:
    """Final research output with enriched context."""
    query: str
    selected_url: str
    selected_title: str
    content: str
    quality_assessment: URLQualityScore
    relevance_justification: str
    search_metadata: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['quality_assessment'] = self.quality_assessment.to_dict()
        return result


class WebResearchPipeline:
    """
    Context-aware web research pipeline that combines search, quality analysis,
    LLM ranking, and intelligent scraping.
    
    Usage:
        async with WebResearchPipeline(llm_callable=my_llm_function) as pipeline:
            result = await pipeline.research("quantum computing applications")
    """
    
    def __init__(
        self,
        llm_callable: Callable[[str], Any],
        use_tor: bool = False,
        quality_threshold: float = 0.3,
        max_urls_to_analyze: int = 10,
        max_urls_for_llm: int = 5,
        scraping_config: Optional[ScrapingConfig] = None
    ):
        """
        Initialize the research pipeline.
        
        Args:
            llm_callable: Async function to call LLM for relevance ranking
            use_tor: Whether to use TOR for anonymous searching
            quality_threshold: Minimum quality score (0.0-1.0) to consider a URL
            max_urls_to_analyze: Maximum URLs to analyze from search results
            max_urls_for_llm: Maximum URLs to send to LLM for ranking
            scraping_config: Custom scraping configuration
        """
        self.llm_callable = llm_callable
        self.use_tor = use_tor
        self.quality_threshold = quality_threshold
        self.max_urls_to_analyze = max_urls_to_analyze
        self.max_urls_for_llm = max_urls_for_llm
        
        # Initialize tools (will be set up in __aenter__)
        self.url_context_tool: Optional[URLContextTool] = None
        self.web_searcher: Optional[WebSearchTool] = None
        self.web_scraper: Optional[WebScraper] = None
        self.scraping_config = scraping_config or ScrapingConfig(
            timeout=30.0,
            max_retries=3,
            enable_pdf_extraction=True
        )
        
        logger.info("WebResearchPipeline initialized")
    
    async def __aenter__(self):
        """Async context manager entry - initialize all tools."""
        logger.info("Setting up research pipeline tools...")
        
        # Initialize URL context tool
        self.url_context_tool = URLContextTool()
        await self.url_context_tool.__aenter__()
        
        # Initialize web searcher
        self.web_searcher = WebSearchTool()
        await self.web_searcher.__aenter__()
        
        # Initialize web scraper
        self.web_scraper = WebScraper(config=self.scraping_config)
        await self.web_scraper.__aenter__()
        
        logger.info("Research pipeline ready")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - cleanup all tools."""
        logger.info("Cleaning up research pipeline...")
        
        if self.url_context_tool:
            await self.url_context_tool.__aexit__(exc_type, exc_val, exc_tb)
        
        if self.web_searcher:
            await self.web_searcher.__aexit__(exc_type, exc_val, exc_tb)
        
        if self.web_scraper:
            await self.web_scraper.__aexit__(exc_type, exc_val, exc_tb)
        
        logger.info("Research pipeline cleanup complete")
    
    def _calculate_quality_score(self, url_analysis: Dict[str, Any]) -> float:
        """
        Calculate normalized quality score from URL analysis.
        
        Args:
            url_analysis: URL analysis result from URLContextTool
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        if 'error' in url_analysis:
            return 0.0
        
        score = 0.0
        
        # Security contributes w1=40%
        security = url_analysis.get('security', {})
        security_score = security.get('score', 0)
        score += (security_score / 100) * 0.4
        
        # Validation contributes w2=20%
        validation = url_analysis.get('validation', {})
        if validation.get('is_valid', False):
            score += 0.2
        
        # Metadata quality contributes w3=20%
        metadata = url_analysis.get('metadata', {})
        if not metadata.get('is_root', True):  # Non-root pages often have more content
            score += 0.1
        if metadata.get('has_query', False):  # Query parameters might indicate dynamic content
            score += 0.05
        if metadata.get('path_depth', 0) > 0:  # Some path depth is good
            score += 0.05
        
        # Connectivity contributes w4=20%
        connectivity = url_analysis.get('connectivity', {})
        if connectivity.get('status') == 'reachable':
            score += 0.2
        
        return min(1.0, max(0.0, score))
    
    async def _analyze_url_quality(
        self,
        search_result: SearchResult
    ) -> URLQualityScore:
        """
        Analyze URL quality using URLContextTool.
        
        Args:
            search_result: Search result to analyze
            
        Returns:
            URLQualityScore with quality assessment
        """
        try:
            # FIX: SearchResult uses 'link' not 'url'
            url = search_result.link
            
            # Analyze URL context
            analysis = self.url_context_tool.analyze_url(url)
            
            # Extract key metrics
            security = analysis.get('security', {})
            parsed = analysis.get('parsed', {})
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(analysis)
            
            return URLQualityScore(
                url=url,
                title=search_result.title,
                snippet=search_result.snippet,
                domain=parsed.get('hostname', urlparse(url).netloc),
                quality_score=quality_score,
                security_score=security.get('score', 0),
                security_level=security.get('level', 'UNKNOWN'),
                is_secure=parsed.get('is_secure', False),
                issues=security.get('issues', []),
                metadata={
                    'search_position': search_result.position,  # FIX: Use 'position' not 'rank'
                    'search_result_type': search_result.result_type,  # FIX: Use 'result_type' not 'engine'
                    'search_source': search_result.source,  # Additional source info
                    'connectivity': analysis.get('connectivity', {}),
                    'url_metadata': analysis.get('metadata', {})
                }
            )
            
        except Exception as e:
            logger.error(f"Error analyzing URL {search_result.link}: {e}")
            return URLQualityScore(
                url=search_result.link,
                title=search_result.title,
                snippet=search_result.snippet,
                domain=urlparse(search_result.link).netloc,
                quality_score=0.0,
                security_score=0,
                security_level='ERROR',
                is_secure=False,
                issues=[f"Analysis failed: {str(e)}"]
            )
    
    async def _rank_urls_with_llm(
        self,
        query: str,
        quality_scored_urls: List[URLQualityScore]
    ) -> tuple[URLQualityScore, str]:
        """
        Use LLM to select the most relevant URL and provide justification.
        
        Args:
            query: Original search query
            quality_scored_urls: List of quality-scored URLs
            
        Returns:
            Tuple of (selected_url_score, justification)
        """
        # Prepare prompt for LLM
        urls_info = []
        for i, url_score in enumerate(quality_scored_urls[:self.max_urls_for_llm], 1):
            urls_info.append(
                f"{i}. URL: {url_score.url}\n"
                f"   Title: {url_score.title}\n"
                f"   Snippet: {url_score.snippet}\n"
                f"   Quality Score: {url_score.quality_score:.2f}\n"
                f"   Security: {url_score.security_level}\n"
                f"   Domain: {url_score.domain}"
            )
        
        prompt = f"""Given the search query: "{query}"

Analyze these top-quality URLs and select THE SINGLE MOST RELEVANT one:

{chr(10).join(urls_info)}

Respond in this exact format:
SELECTED: <number>
JUSTIFICATION: <brief explanation of why this URL is most relevant>

Consider:
1. Semantic alignment with the query
2. Likely depth and quality of information
3. Authority and trustworthiness of the source
4. Recency and specificity of content
"""
        
        try:
            # FIX: Properly await async LLM callable
            if asyncio.iscoroutinefunction(self.llm_callable):
                response = await self.llm_callable(prompt)
            else:
                response = self.llm_callable(prompt)
            
            # Parse response
            if isinstance(response, dict):
                response = response.get('content', str(response))
            response = str(response)
            
            # Extract selection
            selected_idx = None
            justification = "LLM analysis completed"
            
            for line in response.split('\n'):
                line = line.strip()
                if line.startswith('SELECTED:'):
                    try:
                        selected_idx = int(line.split(':', 1)[1].strip()) - 1
                    except (ValueError, IndexError):
                        pass
                elif line.startswith('JUSTIFICATION:'):
                    justification = line.split(':', 1)[1].strip()
            
            # Validate selection
            if selected_idx is None or selected_idx < 0 or selected_idx >= len(quality_scored_urls):
                logger.warning(f"LLM returned invalid selection (idx={selected_idx}), using top-ranked URL")
                return quality_scored_urls[0], "Selected as top-ranked result (LLM selection invalid)"
            
            return quality_scored_urls[selected_idx], justification
            
        except Exception as e:
            logger.error(f"Error in LLM ranking: {e}")
            return quality_scored_urls[0], f"Selected as top-ranked result (LLM error: {str(e)})"
    
    async def research(
        self,
        query: str,
        num_results: int = 10,
        search_kwargs: Optional[Dict[str, Any]] = None
    ) -> ResearchResult:
        """
        Execute the complete research pipeline.
        
        Args:
            query: Search query
            num_results: Number of search results to retrieve
            search_kwargs: Additional arguments for web search
            
        Returns:
            ResearchResult with enriched content and metadata
        """
        logger.info(f"Starting research pipeline for query: '{query}'")
        
        # Step 1: Perform web search
        logger.info("Step 1: Performing web search...")
        search_kwargs = search_kwargs or {}
        
        # FIX: Ensure parse_result_for_llm is set correctly
        search_response: SearchResponse = await self.web_searcher.search(
            query,
            num_results=num_results,
            parse_result_for_llm=False,  # We want structured results, not formatted string
            **search_kwargs
        )
        
        if not search_response.results:
            raise ValueError(f"Web search returned zero results for query: '{query}'")
        
        logger.info(f"Found {len(search_response.results)} search results")
        
        # Step 2: Analyze URL quality for top results
        logger.info("Step 2: Analyzing URL quality...")
        urls_to_analyze = search_response.results[:self.max_urls_to_analyze]
        
        quality_scores = await asyncio.gather(
            *[self._analyze_url_quality(result) for result in urls_to_analyze],
            return_exceptions=True
        )
        
        # Filter out exceptions and low-quality URLs
        valid_quality_scores = [
            score for score in quality_scores
            if isinstance(score, URLQualityScore) and score.quality_score >= self.quality_threshold
        ]
        
        if not valid_quality_scores:
            logger.warning(f"No URLs met quality threshold of {self.quality_threshold}")
            # Fallback: use all scores and lower threshold
            valid_quality_scores = [
                score for score in quality_scores
                if isinstance(score, URLQualityScore)
            ]
            if not valid_quality_scores:
                raise ValueError("No valid URLs found after quality analysis")
        
        logger.info(f"Found {len(valid_quality_scores)} URLs meeting quality threshold")
        
        # Step 3: Rank by quality score
        logger.info("Step 3: Ranking URLs by quality...")
        valid_quality_scores.sort(key=lambda x: x.quality_score, reverse=True)
        
        # Step 4: Use LLM to select most relevant URL
        logger.info("Step 4: Consulting LLM for relevance ranking...")
        selected_url_score, justification = await self._rank_urls_with_llm(
            query,
            valid_quality_scores
        )
        
        logger.info(f"LLM selected: {selected_url_score.url}")
        logger.info(f"Justification: {justification}")

        # Rate limiting between LLM call and scraping
        await asyncio.sleep(2.5)
        
        # Step 5: Scrape the selected URL
        logger.info("Step 5: Scraping selected URL...")
        try:
            content = await self.web_scraper.scrape_text(
                selected_url_score.url,
                clean_text=True
            )
            
            if not content or len(content.strip()) < 50:
                logger.warning(f"Scraping returned minimal content: {len(content)} chars")
                content = f"[Scraping returned limited content]\n\nTitle: {selected_url_score.title}\nSnippet: {selected_url_score.snippet}"
        except Exception as e:
            logger.error(f"Error scraping URL: {e}")
            content = f"[Error scraping content: {str(e)}]\n\nTitle: {selected_url_score.title}\nSnippet: {selected_url_score.snippet}"
        
        # Step 6: Build and return enriched result
        logger.info("Step 6: Building final result...")
        result = ResearchResult(
            query=query,
            selected_url=selected_url_score.url,
            selected_title=selected_url_score.title,
            content=content,
            quality_assessment=selected_url_score,
            relevance_justification=justification,
            search_metadata={
                'total_results_found': len(search_response.results),
                'urls_analyzed': len(urls_to_analyze),
                'urls_meeting_quality': len(valid_quality_scores),
                'search_time': search_response.search_time,
                'top_quality_urls': [
                    {
                        'url': score.url,
                        'title': score.title,
                        'quality_score': score.quality_score,
                        'security_level': score.security_level
                    }
                    for score in valid_quality_scores[:5]
                ]
            }
        )
        
        logger.info("Research pipeline completed successfully")
        return result


# Convenience function for one-off research tasks
async def quick_research(
    query: str,
    llm_callable: Callable[[str], Any],
    use_tor: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function for quick research tasks.
    
    Args:
        query: Search query
        llm_callable: Async or sync function to call LLM
        use_tor: Whether to use TOR
        **kwargs: Additional pipeline configuration
        
    Returns:
        Research result as dictionary
    """
    async with WebResearchPipeline(llm_callable=llm_callable, use_tor=use_tor, **kwargs) as pipeline:
        result = await pipeline.research(query)
        return result.to_dict()