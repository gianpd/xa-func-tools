"""
Advanced AI Research Agent - Simplified and Improved
Intelligent research automation with premium search and AI analysis.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from src.function_calling import FunctionCalling
from src.function_calling.tools.research_pipeline import WebResearchPipeline, ResearchResult, SearchResult
from src.function_calling.core import call_llm

# Configure comprehensive logging (restored from original for better observability)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ResearchSource:
    """Data structure for storing analyzed research sources."""
    title: str
    url: str
    content: str
    summary: str
    credibility_score: float
    security_score: int
    relevance_score: float
    key_points: List[str]
    word_count: int # Restored for consistency
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'title': self.title,
            'url': self.url,
            'content_preview': self.content[:300] + '...' if len(self.content) > 300 else self.content,
            'summary': self.summary,
            'credibility_score': self.credibility_score,
            'security_score': self.security_score,
            'relevance_score': self.relevance_score,
            'key_points': self.key_points,
            'word_count': self.word_count,
            'metadata': self.metadata
        }


@dataclass 
class ResearchReport:
    """Comprehensive research report structure."""
    request: str
    generated_queries: List[str]
    sources_analyzed: int
    total_search_results: int
    research_time: float
    key_findings: List[str]
    executive_summary: str
    detailed_analysis: str
    sources: List[ResearchSource]
    confidence_score: float
    limitations: List[str]
    recommendations: List[str]
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'request': self.request,
            'generated_queries': self.generated_queries,
            'sources_analyzed': self.sources_analyzed,
            'total_search_results': self.total_search_results,
            'research_time': self.research_time,
            'key_findings': self.key_findings,
            'executive_summary': self.executive_summary,
            'detailed_analysis': self.detailed_analysis,
            'sources': [source.to_dict() for source in self.sources],
            'confidence_score': self.confidence_score,
            'limitations': self.limitations,
            'recommendations': self.recommendations,
            'created_at': self.created_at
        }
    

class PremiumResearchAgent:
    """Simplified AI research agent leveraging AI for most analysis tasks."""
    
    def __init__(self, 
                 ai_model: str = "qwen/qwen3-30b-a3b-thinking-2507",
                 max_queries: int = 5,
                 max_sources: int = 3,
                 min_security_score: int = 60,
                 min_credibility_score: float = 0.3): # Restored min_credibility_score
        self.ai_model = ai_model
        self.max_queries = max_queries
        self.max_sources = max_sources
        self.min_security_score = min_security_score
        self.min_credibility_score = min_credibility_score # Stored for programmatic use
        
        # Single AI agent instance
        self.agent = FunctionCalling(
            model=ai_model,
            max_turns=25,
            system_prompt=self._create_system_prompt(),
            temperature=0.1,
            enable_logging=True
        )
        
        # Statistics tracking (Restored for enterprise features)
        self.stats = {
            'total_requests': 0,
            'total_queries_generated': 0,
            'total_searches_performed': 0,
            'total_search_results_gathered': 0,
            'total_sources_analyzed': 0,
            'avg_research_time': 0.0,
            'security_rejections': 0,
            'credibility_rejections': 0
        }

        # Create directories
        Path("logs").mkdir(exist_ok=True)
        Path("reports").mkdir(exist_ok=True)
    
    def _create_system_prompt(self) -> str:
        return f"""You are an expert AI research analyst. Execute comprehensive research using available tools:

**RESEARCH WORKFLOW:**
1. **QUERY GENERATION**: Create up to {self.max_queries} strategic search queries from the user's request.
2. **SEARCH EXECUTION**: Use the `search` tool for each query, collecting all relevant results.
3. **ANALYSIS & SYNTHESIS**: Analyze the extracted content. For each *selected* source, identify its title, summarize its content, extract key points, and assess its credibility (0.0-1.0) and relevance (0.0-1.0) to the original research request. Synthesize all information to form key findings, an executive summary, a detailed analysis, limitations, and recommendations.

**TOOL USAGE:**
- search(query: str, num_results: int = 10): Execute web searches and return a list of search results.

**OUTPUT REQUIREMENTS:**
After completing all research steps, provide the comprehensive analysis in this JSON format:
{{
    "queries_generated": ["query1", "query2", ...],
    "search_results_count": total_number_of_raw_search_results_found,
    "sources_analyzed": [
        {{
            "title": "extracted_title",
            "url": "source_url", 
            "summary": "content_summary",
            "credibility_assessment": score_0_to_1,
            "relevance_score": score_0_to_1,
            "key_points": ["point1", "point2", "point3"],
            "security_score": extracted_from_url_context,
            "content_length": word_count_of_scraped_content
        }}
    ],
    "key_findings": ["finding1", "finding2", ...],
    "executive_summary": "comprehensive_overview",
    "detailed_analysis": "thorough_analysis_with_citations",
    "confidence_score": overall_confidence_0_to_1,
    "limitations": ["limitation1", "limitation2"],
    "recommendations": ["rec1", "rec2", "rec3"]
}}

**QUALITY STANDARDS:**
- Prioritize authoritative domains (.edu, .gov, established organizations) when selecting sources.
- Extract meaningful insights and synthesize across sources.
- Provide evidence-based conclusions with proper attribution (e.g., [Source 1], [Source 2]).
- Acknowledge limitations and potential biases transparently.
- Ensure all scores (credibility, relevance, confidence) are float values between 0.0 and 1.0."""
    
    async def conduct_research(self, user_request: str) -> ResearchReport:
        """Main research execution method."""
        start_time = time.time()
        self.stats['total_requests'] += 1 # Update statistics
        logger.info(f"🔬 Starting research: '{user_request}'")
        
        try:
            async with (
               WebResearchPipeline(llm_callable=call_llm) as searcher
            ):
                # Register tools
                self.agent.register_tool(searcher.research)
                
                # Execute research
                final_answer, execution_log = await self.agent.run_async(
                    f"Conduct comprehensive research on: {user_request}",
                    tool_choice="auto",
                )
                
                # Parse results and generate report
                report = await self._parse_and_generate_report(
                    user_request, final_answer, execution_log, time.time() - start_time
                )
                
                await self._save_report(report)
                logger.info(f"🔬 Research completed in {time.time() - start_time:.2f}s")
                return report
                
        except Exception as e:
            logger.error(f"🔬 Research failed: {e}", exc_info=True)
            return self._create_error_report(user_request, str(e), time.time() - start_time)
    
    async def _parse_and_generate_report(self, user_request: str, final_answer: str, 
                                       execution_log: List[Dict], research_time: float) -> ResearchReport:
        """Parse AI output and execution log to generate structured report."""
        
        # Extract JSON from AI response
        analysis_data = self._extract_json_from_response(final_answer)
        
        # Parse execution log for actual tool results (updated to pass self.stats)
        execution_data = self._parse_execution_log(execution_log, self.stats)
        
        sources: List[SearchResult] = []
        for source_data in analysis_data.get('sources_analyzed', []):
            try:
                url = source_data.get('url', '')
                actual_content = execution_data['scraped_content'].get(url, '')
                actual_url_context = execution_data['url_contexts'].get(url, {})
                actual_security_score = actual_url_context.get('security_score', 0)
                
                # Programmatic enforcement of min_security_score (already in system prompt for AI)
                if actual_security_score < self.min_security_score:
                    self.stats['security_rejections'] += 1
                    logger.debug(f"Source {url} rejected due to low security score ({actual_security_score}).")
                    continue
                
                # Programmatic enforcement of min_credibility_score (Restored crucial point)
                ai_credibility_score = float(source_data.get('credibility_assessment', 0.0))
                if ai_credibility_score < self.min_credibility_score:
                    self.stats['credibility_rejections'] += 1
                    logger.debug(f"Source {url} rejected due to low AI credibility score ({ai_credibility_score}).")
                    continue

                if actual_content: # Ensure we have content
                    source = ResearchResult(
                        title=source_data.get('title', 'Unknown Title'),
                        url=url,
                        content=actual_content,
                        summary=source_data.get('summary', 'No summary provided.'),
                        credibility_score=ai_credibility_score,
                        security_score=actual_security_score,
                        relevance_score=float(source_data.get('relevance_score', 0.0)),
                        key_points=source_data.get('key_points', []),
                        word_count=len(actual_content.split()), # Populate word_count
                        metadata={
                            'ai_analysis_provided': True,
                            'full_url_context': actual_url_context.get('full_context') # Store raw context
                        }
                    )
                    sources.append(source)
            except (ValueError, TypeError, KeyError) as e:
                logger.warning(f"🔬 Error processing AI-analyzed source for URL '{source_data.get('url')}': {e}. Skipping source.")
                continue
        
        # Update stats
        self.stats['total_sources_analyzed'] += len(sources)

        # Create final report
        return ResearchReport(
            request=user_request,
            generated_queries=analysis_data.get('queries_generated', []),
            sources_analyzed=len(sources),
            total_search_results=execution_data['total_search_results'], # Already captured in stats, but also in report
            research_time=research_time,
            key_findings=analysis_data.get('key_findings', []),
            executive_summary=analysis_data.get('executive_summary', ''),
            detailed_analysis=analysis_data.get('detailed_analysis', ''),
            sources=sources,
            confidence_score=float(analysis_data.get('confidence_score', 0.5)), # Ensure float
            limitations=analysis_data.get('limitations', []),
            recommendations=analysis_data.get('recommendations', [])
        )
    
    def _extract_json_from_response(self, response: str) -> Dict[str, Any]:
        """Extract JSON data from AI response."""
        try:
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
        except json.JSONDecodeError as e:
            logger.warning(f"🔬 JSON parsing failed: {e}. Attempting partial extraction or fallback.")
        except Exception as e:
            logger.warning(f"🔬 Error during JSON extraction: {e}")
        
        logger.info(f"🔬 Falling back to non-JSON parsing for AI response.")
        # Return a minimal structure if parsing fails, trying to infer from raw string
        return {
            'queries_generated': [],
            'sources_analyzed': [],
            'key_findings': [f'AI analysis performed, but structured JSON output could not be fully parsed. Raw response: {response[:150]}...'],
            'executive_summary': response[:500] + '...' if len(response) > 500 else response,
            'detailed_analysis': response,
            'confidence_score': 0.1, # Low confidence if parsing failed
            'limitations': ['Automated JSON parsing of AI output failed or was incomplete.'],
            'recommendations': ['Review AI raw response for insights.']
        }
    
    def _parse_execution_log(self, execution_log: List[Dict], stats: Dict[str, Any]) -> Dict[str, Any]:
        """Extract actual tool results from execution log and update statistics."""
        data = {
            'search_results': [],
            'url_contexts': {},
            'scraped_content': {},
            'total_search_results': 0 # This will be the actual count from all search tools
        }
        
        for turn in execution_log:
            tool_calls = turn.get('tool_calls', [])
            tool_results = turn.get('tool_results', [])
            
            for i, tool_call in enumerate(tool_calls):
                function_name = tool_call.get('function', {}).get('name', '')
                
                try:
                    # Args are usually JSON strings, results can be dicts or strings
                    args = json.loads(tool_call.get('function', {}).get('arguments', '{}'))
                    result = tool_results[i] if i < len(tool_results) else None
                    
                    if function_name == 'research':
                        stats['total_searches_performed'] += 1 # Update stats
                        if isinstance(result, dict) and 'results' in result:
                            data['search_results'].extend(result['results'])
                            data['total_search_results'] += len(result['results'])
                            stats['total_search_results_gathered'] += len(result['results']) # Update stats
                    
                    elif function_name == 'get_full_context':
                        url = args.get('url', '')
                        if url and result:
                            # URLContextTool is expected to return a dict, if it's a string, try to parse or log
                            if isinstance(result, str):
                                try:
                                    parsed_result = json.loads(result)
                                except json.JSONDecodeError:
                                    logger.warning(f"URLContextTool returned string for {url}, not JSON: {result[:100]}...")
                                    parsed_result = {'security_score': 0, 'full_context': result} # Fallback
                            else: # Assume it's already a dict
                                parsed_result = result
                            
                            security_score = parsed_result.get('security_score', parsed_result.get('security', {}).get('score', 0))
                            data['url_contexts'][url] = {
                                'security_score': security_score,
                                'full_context': parsed_result
                            }
                    
                    elif function_name == 'scrape_text':
                        url = args.get('url', '')
                        if url and result and isinstance(result, str):
                            data['scraped_content'][url] = result
                
                except Exception as e:
                    logger.debug(f"🔬 Error parsing tool call or result for '{function_name}': {e}. Turn: {turn}. Result: {result}")
                    continue
        
        logger.info(f"🔬 Execution summary: {data['total_search_results']} search results, "
                   f"{len(data['url_contexts'])} URLs validated, {len(data['scraped_content'])} contents extracted")
        
        return data
    
    def _create_error_report(self, user_request: str, error_msg: str, research_time: float) -> ResearchReport:
        """Create error report when research fails."""
        logger.error(f"Creating error report for request '{user_request}' due to: {error_msg}")
        return ResearchReport(
            request=user_request,
            generated_queries=[],
            sources_analyzed=0,
            total_search_results=0,
            research_time=research_time,
            key_findings=[f"Research failed: {error_msg}"],
            executive_summary=f"Research could not be completed successfully due to a critical error: {error_msg}. Please review logs for details.",
            detailed_analysis="",
            sources=[],
            confidence_score=0.0,
            limitations=["Research incomplete due to technical error", "No sources analyzed"],
            recommendations=["Please check system logs and retry the research request."]
        )
    
    async def _save_report(self, report: ResearchReport):
        """Save research report to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Sanitize user_request for filename (from original version)
        sanitized_request = "".join(c if c.isalnum() else "_" for c in report.request)[:50]
        filename = f"reports/research_report_{timestamp}_{sanitized_request}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
            logger.info(f"🔬 Report saved: {filename}")
        except Exception as e:
            logger.error(f"🔬 Failed to save report: {e}")

async def main():
    """Demo the simplified research agent."""
    print("🔬 Premium AI Research Agent - Simplified")
    print("=" * 50)
    
    agent = PremiumResearchAgent(
        max_queries=5,
        max_sources=3,
        min_security_score=60,
        min_credibility_score=0.3 # Passed to agent for internal use
    )
    
    research_topics = [
        "Impact of artificial intelligence on employment trends in 2024",
        "Renewable energy investment opportunities and market outlook", 
        "Cybersecurity threats and best practices for small businesses"
    ]
    
    print("Available research topics:")
    for i, topic in enumerate(research_topics, 1):
        print(f"{i}. {topic}")
    print(f"{len(research_topics) + 1}. Custom topic")
    
    try:
        choice = input("\nSelect topic (1-4) or press Enter for demo: ").strip()
        
        if choice == "" or choice == "1":
            user_request = research_topics[0]
        elif choice.isdigit() and 1 <= int(choice) <= len(research_topics):
            user_request = research_topics[int(choice) - 1]
        elif choice == str(len(research_topics) + 1):
            user_request = input("Enter custom research request: ").strip()
            if not user_request:
                user_request = research_topics[0]
        else:
            user_request = research_topics[0]
        
        print(f"\n🔬 Researching: '{user_request}'")
        print("=" * 50)
        
        # Conduct research
        report = await agent.conduct_research(user_request)
        
        # Display results
        print(f"\n📋 RESEARCH RESULTS")
        print("=" * 50)
        print(f"📝 Request: {report.request}")
        print(f"⏱️  Time: {report.research_time:.2f}s")
        print(f"🔍 Queries Generated: {len(report.generated_queries)}") # Updated label
        print(f"📊 Sources Analyzed: {report.sources_analyzed}") # Updated label
        print(f"🎯 Confidence: {report.confidence_score:.2%}")
        
        if report.generated_queries:
            print(f"\n📋 SEARCH QUERIES:")
            for i, query in enumerate(report.generated_queries, 1):
                print(f"  {i}. {query}")
        
        print(f"\n🔑 KEY FINDINGS:")
        for i, finding in enumerate(report.key_findings, 1):
            print(f"  {i}. {finding}")
        
        print(f"\n📖 EXECUTIVE SUMMARY:")
        print(f"{report.executive_summary}")
        
        if report.sources:
            print(f"\n📚 ANALYZED SOURCES:")
            for i, source in enumerate(report.sources, 1):
                print(f"\n{i}. {source.title}")
                print(f"   🔗 {source.url}")
                print(f"   ⭐ Credibility: {source.credibility_score:.2f}")
                print(f"   🔒 Security: {source.security_score}")
                print(f"   📊 Relevance: {source.relevance_score:.2f}")
                print(f"   📝 Words: {source.word_count:,}") # Display word count
                if source.key_points:
                    print(f"   💡 Key: {source.key_points[0][:80]}...")
                print(f"   💬 Summary: {source.summary}") # Display summary
        
        print(f"\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(report.recommendations, 1):
            print(f"  {i}. {rec}")

        if report.limitations:
            print(f"\n⚠️  LIMITATIONS:")
            for i, limitation in enumerate(report.limitations, 1):
                print(f"  {i}. {limitation}")

        # Display statistics (restored for enterprise features)
        stats = agent.stats
        print(f"\n📈 AGENT STATISTICS:")
        print(f"  • Total Requests: {stats['total_requests']}")
        print(f"  • Queries Generated: {stats['total_queries_generated']}")
        print(f"  • Searches Performed: {stats['total_searches_performed']}")
        print(f"  • Total Search Results Gathered: {stats['total_search_results_gathered']}")
        print(f"  • Sources Analyzed: {stats['total_sources_analyzed']}")
        print(f"  • Security Rejections: {stats['security_rejections']}")
        print(f"  • Credibility Rejections: {stats['credibility_rejections']}")

        print(f"\n💾 Full report saved to reports/ directory")
        
    except KeyboardInterrupt:
        print("\n🛑 Interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        logger.error(f"Main execution error: {e}", exc_info=True)

if __name__ == "__main__":
    # Ensure event loop policy is set for Windows compatibility (from original version)
    if hasattr(asyncio, 'WindowsProactorEventLoopPolicy'):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(main())