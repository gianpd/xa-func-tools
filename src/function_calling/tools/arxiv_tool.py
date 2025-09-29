#!/usr/bin/env python3
"""
ArXiv Tool - A function tool for searching and retrieving arXiv papers
Designed for use in agent systems and function calling
"""

import arxiv
import time
import logging
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import json
import re


class ArxivTool:
    """Tool for searching and retrieving arXiv papers via function calls."""
    
    def __init__(self):
        self.client = arxiv.Client()
        self.logger = logging.getLogger(__name__)
        
        # Rate limiting settings
        self.requests_per_burst = 4
        self.burst_delay = 1.0  # seconds
        self.request_delay = 0.25  # seconds between requests
        self.request_count = 0
        self.last_request_time = 0
        
    def _rate_limit(self):
        """Apply rate limiting to requests."""
        current_time = time.time()
        
        # Check if we need a burst delay
        if self.request_count > 0 and self.request_count % self.requests_per_burst == 0:
            if current_time - self.last_request_time < self.burst_delay:
                time.sleep(self.burst_delay)
        
        # Small delay between all requests
        elif current_time - self.last_request_time < self.request_delay:
            time.sleep(self.request_delay)
            
        self.last_request_time = time.time()
        self.request_count += 1
    
    def search_papers(self, 
                     query: str, 
                     max_results: int = 10,
                     sort_by: str = "submittedDate",
                     sort_order: str = "descending") -> Dict:
        """
        Search for papers on arXiv.
        
        Args:
            query: Search query (e.g., "quantum computing", "cat:cs.AI", "au:smith")
            max_results: Maximum number of results to return (default: 10, max: 2000)
            sort_by: Sort criterion ("submittedDate", "lastUpdatedDate", "relevance")
            sort_order: Sort order ("ascending", "descending")
            
        Returns:
            Dict with search results and metadata
        """
        try:
            self._rate_limit()
            
            # Validate inputs
            max_results = min(max_results, 2000)  # arXiv API limit
            
            sort_criteria = {
                "submittedDate": arxiv.SortCriterion.SubmittedDate,
                "lastUpdatedDate": arxiv.SortCriterion.LastUpdatedDate,
                "relevance": arxiv.SortCriterion.Relevance
            }
            
            sort_orders = {
                "ascending": arxiv.SortOrder.Ascending,
                "descending": arxiv.SortOrder.Descending
            }
            
            # Create search
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=sort_criteria.get(sort_by, arxiv.SortCriterion.SubmittedDate),
                sort_order=sort_orders.get(sort_order, arxiv.SortOrder.Descending)
            )
            
            # Execute search
            results = list(self.client.results(search))
            
            # Format results
            papers = []
            for paper in results:
                papers.append(self._format_paper(paper))
            
            return {
                "success": True,
                "query": query,
                "total_results": len(papers),
                "papers": papers,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Search failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "query": query,
                "timestamp": datetime.now().isoformat()
            }
    
    def get_paper_by_id(self, arxiv_id: str) -> Dict:
        """
        Get a specific paper by its arXiv ID.
        
        Args:
            arxiv_id: arXiv ID (e.g., "1706.03762" or "arxiv:1706.03762")
            
        Returns:
            Dict with paper details
        """
        try:
            self._rate_limit()
            
            # Clean the ID
            arxiv_id = arxiv_id.replace("arxiv:", "")
            
            # Search by ID
            search = arxiv.Search(id_list=[arxiv_id])
            results = list(self.client.results(search))
            
            if not results:
                return {
                    "success": False,
                    "error": f"Paper with ID {arxiv_id} not found",
                    "arxiv_id": arxiv_id
                }
            
            paper = results[0]
            
            return {
                "success": True,
                "paper": self._format_paper(paper, include_full_summary=True),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Get paper by ID failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "arxiv_id": arxiv_id,
                "timestamp": datetime.now().isoformat()
            }
    
    def search_by_category(self, 
                          category: str, 
                          max_results: int = 20,
                          recent_days: Optional[int] = None) -> Dict:
        """
        Search papers by arXiv category.
        
        Args:
            category: arXiv category (e.g., "cs.AI", "cs.LG", "math.NA")
            max_results: Maximum number of results
            recent_days: Only return papers from last N days (optional)
            
        Returns:
            Dict with papers from the category
        """
        try:
            query = f"cat:{category}"
            
            # Add date filter if specified
            if recent_days:
                cutoff_date = datetime.now() - timedelta(days=recent_days)
                date_str = cutoff_date.strftime("%Y%m%d")
                query += f" AND submittedDate:[{date_str}* TO *]"
            
            return self.search_papers(
                query=query,
                max_results=max_results,
                sort_by="submittedDate"
            )
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "category": category,
                "timestamp": datetime.now().isoformat()
            }
    
    def search_by_author(self, author_name: str, max_results: int = 20) -> Dict:
        """
        Search papers by author name.
        
        Args:
            author_name: Author name (e.g., "Geoffrey Hinton")
            max_results: Maximum number of results
            
        Returns:
            Dict with papers by the author
        """
        try:
            # Format author query
            query = f'au:"{author_name}"'
            
            return self.search_papers(
                query=query,
                max_results=max_results,
                sort_by="submittedDate"
            )
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "author": author_name,
                "timestamp": datetime.now().isoformat()
            }
    
    def get_recent_papers(self, 
                         category: Optional[str] = None,
                         days: int = 7,
                         max_results: int = 50) -> Dict:
        """
        Get recent papers, optionally filtered by category.
        
        Args:
            category: arXiv category filter (optional)
            days: Number of days to look back (default: 7)
            max_results: Maximum number of results
            
        Returns:
            Dict with recent papers
        """
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Format dates for arXiv API
            start_str = start_date.strftime("%Y%m%d")
            end_str = end_date.strftime("%Y%m%d")
            
            # Build query
            query = f"submittedDate:[{start_str}* TO {end_str}*]"
            if category:
                query = f"cat:{category} AND " + query
            
            return self.search_papers(
                query=query,
                max_results=max_results,
                sort_by="submittedDate"
            )
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "category": category,
                "days": days,
                "timestamp": datetime.now().isoformat()
            }
    
    def _format_paper(self, paper, include_full_summary: bool = False) -> Dict:
        """Format a paper object into a clean dictionary."""
        try:
            # Get summary - truncate if not full version
            summary = paper.summary
            if not include_full_summary and len(summary) > 500:
                summary = summary[:500] + "..."
            
            return {
                "id": paper.get_short_id(),
                "title": paper.title,
                "authors": [author.name for author in paper.authors],
                "summary": summary,
                "categories": paper.categories,
                "primary_category": paper.primary_category,
                "published": paper.published.isoformat() if paper.published else None,
                "updated": paper.updated.isoformat() if paper.updated else None,
                "pdf_url": paper.pdf_url,
                "arxiv_url": paper.entry_id,
                "comment": getattr(paper, 'comment', None),
                "journal_ref": getattr(paper, 'journal_ref', None)
            }
            
        except Exception as e:
            self.logger.error(f"Error formatting paper: {str(e)}")
            return {
                "id": "unknown",
                "title": "Error formatting paper",
                "error": str(e)
            }
    
    def get_paper_summary(self, arxiv_id: str) -> Dict:
        """
        Get just the summary/abstract of a specific paper.
        
        Args:
            arxiv_id: arXiv ID
            
        Returns:
            Dict with paper summary
        """
        try:
            paper_data = self.get_paper_by_id(arxiv_id)
            
            if not paper_data.get("success"):
                return paper_data
            
            paper = paper_data["paper"]
            
            return {
                "success": True,
                "id": paper["id"],
                "title": paper["title"],
                "authors": paper["authors"],
                "summary": paper["summary"],
                "categories": paper["categories"],
                "published": paper["published"],
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "arxiv_id": arxiv_id,
                "timestamp": datetime.now().isoformat()
            }


# Function interfaces for tool calling
def search_arxiv_papers(query: str, max_results: int = 10, sort_by: str = "submittedDate") -> Dict:
    """Search arXiv papers by query."""
    tool = ArxivTool()
    return tool.search_papers(query, max_results, sort_by)


def get_arxiv_paper(arxiv_id: str) -> Dict:
    """Get a specific arXiv paper by ID."""
    tool = ArxivTool()
    return tool.get_paper_by_id(arxiv_id)


def search_arxiv_by_category(category: str, max_results: int = 20, recent_days: Optional[int] = None) -> Dict:
    """Search arXiv papers by category."""
    tool = ArxivTool()
    return tool.search_by_category(category, max_results, recent_days)


def search_arxiv_by_author(author_name: str, max_results: int = 20) -> Dict:
    """Search arXiv papers by author."""
    tool = ArxivTool()
    return tool.search_by_author(author_name, max_results)


def get_recent_arxiv_papers(category: Optional[str] = None, days: int = 7, max_results: int = 50) -> Dict:
    """Get recent arXiv papers."""
    tool = ArxivTool()
    return tool.get_recent_papers(category, days, max_results)


def get_arxiv_paper_summary(arxiv_id: str) -> Dict:
    """Get summary of a specific arXiv paper."""
    tool = ArxivTool()
    return tool.get_paper_summary(arxiv_id)


# Example usage and testing
if __name__ == "__main__":
    # Test the tool
    print("Testing ArXiv Tool...")
    
    # Test 1: Search papers
    print("\n1. Searching for quantum computing papers:")
    result = search_arxiv_papers("quantum computing", max_results=3)
    if result["success"]:
        for paper in result["papers"]:
            print(f"  - {paper['title'][:60]}...")
    else:
        print(f"  Error: {result['error']}")
    
    # Test 2: Get paper by ID
    print("\n2. Getting specific paper (Attention Is All You Need):")
    result = get_arxiv_paper("1706.03762")
    if result["success"]:
        paper = result["paper"]
        print(f"  Title: {paper['title']}")
        print(f"  Authors: {', '.join(paper['authors'][:3])}...")
    else:
        print(f"  Error: {result['error']}")
    
    # Test 3: Search by category
    print("\n3. Recent AI papers:")
    result = search_arxiv_by_category("cs.AI", max_results=3, recent_days=30)
    if result["success"]:
        print(f"  Found {result['total_results']} papers")
        for paper in result["papers"]:
            print(f"  - {paper['title'][:60]}...")
    else:
        print(f"  Error: {result['error']}")
    
    # Test 4: Search by author
    print("\n4. Papers by Geoffrey Hinton:")
    result = search_arxiv_by_author("Geoffrey Hinton", max_results=2)
    if result["success"]:
        print(f"  Found {result['total_results']} papers")
        for paper in result["papers"]:
            print(f"  - {paper['title'][:60]}...")
    else:
        print(f"  Error: {result['error']}")
    
    print("\nTesting complete!")