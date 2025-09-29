"""
Tools submodule for the function calling framework.

This module provides pre-built tools for web scraping, search, URL analysis, and research.
All tools are designed to be registered with FunctionCalling instances.

@dependencies
- WebScraper: From web_scraper.py (requires aiohttp, beautifulsoup4, markitdown for PDF)
- URLContextTool: From url_context.py (standard library only)
- SerperLikeWebSearcher: From web_search.py (requires httpx, bs4, ddgs, stem for TOR)
- ArxivTool: From arxiv_tool.py (requires arxiv library)

@notes
- Tools are async where possible for non-blocking execution.
- Error handling: Tools return structured dicts with 'success' and 'error' fields.
- Security: URLContextTool validates security before scraping; TOR support in search tools.
"""

from .web_scraper import WebScraper
from .url_context import URLContextTool
from .arxiv_tool import ArxivTool
from .web_search import WebSearchTool
from .web_browser import WebBrowser


# Define public API for import *
__all__ = [
    "WebScraper",
    "WebBrowser",
    "URLContextTool", 
    "WebSearchTool",
    "ArxivTool",
]

__version__ = "1.0.0"
__author__ = "gianpd"
__license__ = "MIT"