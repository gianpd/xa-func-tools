"""
Automated SEO Competitor Analysis

4-step workflow (search → validate → scrape → analyze)

Demonstrates how to:
1. Search for top-ranking content on a topic
2. Validate URLs for security and credibility
3. Extract and synthesize competitor content
4. Generate structured insights with AI

This showcases xa-func-tools' ability to chain web search,
scraping, and LLM analysis in a practical workflow.

================================================================================
💡 SEO Agent Demo: Automated Competitor Research 💡
================================================================================
This script demonstrates how to analyze top-ranking competitors for a given topic to inform SEO strategy.
2025-09-29 22:04:26,620 - INFO - 🚀 Starting competitor analysis for topic: 'Home made gadgets'
2025-09-29 22:04:26,621 - INFO - Step 1: Searching for top competitors...
2025-09-29 22:04:26,881 - INFO - Found 4 potential competitors.
2025-09-29 22:04:26,881 - INFO - Step 2: Validating and scraping content from the top 3 credible sources...
2025-09-29 22:04:27,097 - INFO -   ✅ URL 'https://www.zdnet.com/home-and-office/i-stock-my-toolkit-with-these-10-diy-gadgets-and-theyre-all-i-need/' is credible (Score: 100). Scraping content...
2025-09-29 22:04:30,296 - INFO -   ✅ URL 'https://www.reddit.com/r/diyelectronics/comments/19djbtr/what_are_some_cool_gadgets_that_absolute/' is credible (Score: 100). Scraping content...
2025-09-29 22:04:31,391 - INFO -   ✅ URL 'https://interestingengineering.com/lists/7-homemade-gadgets-to-inspire-the-inventor-within-you' is credible (Score: 100). Scraping content...
2025-09-29 22:04:33,063 - INFO - Successfully scraped content from 3 competitors.
2025-09-29 22:04:33,063 - INFO - Step 3: Generating final analysis report with AI...

--- Agent Turn 1 ---
2025-09-29 22:04:36,259 - INFO - HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 200 OK"
🤖 Agent's Thought: 

{
    "topic_analyzed": "Home made gadgets",
    "competitors_analyzed": [
        "https://www.zdnet.com/home-and-office/i-stock-my-toolkit-with-these-10-diy-gadgets-and-theyre-all-i-need/",
        "https://www.reddit.com/r/diyelectronics/comments/19djbtr/what_are_some_cool_gadgets_that_absolute/",
        "https://interestingengineering.com/lists/7-homemade-gadgets-to-inspire-the-inventor-within-you"
    ],
    "common_keywords": [
        "DIY",
        "gadgets",
        "tutorial",
        "beginner",
        "tools"
    ],
    "content_structure_insights": [
        "Most articles structure content as numbered or bullet-point lists with detailed gadget descriptions and sourcing information",
        "Step-by-step instructions and purchasing links are commonly included, often accompanied by images and external references"
    ],
    "key_themes_and_angles": [
        "Professional tool recommendations for home improvement (ZDNET) vs. beginner electronics projects (Reddit) vs. creative repurposing of household items (IE)",
        "Emphasis on low-cost, accessible materials and practical solutions for DIY enthusiasts"
    ],
    "executive_summary": "The competitive landscape shows diverse approaches from professional tool reviews to beginner tutorials and creative DIY projects. To stand out, create content that combines clear step-by-step guides with cost-effective, safety-conscious solutions, addressing specific user needs like accessibility for beginners and durable, multi-use gadgets."
}
✅ Agent finished.
2025-09-29 22:05:01,972 - INFO - Step 4: Finalizing the report.


✅ Analysis Complete! Here is your report:

{
  "topic_analyzed": "Home made gadgets",
  "competitors_analyzed": [
    "https://www.zdnet.com/home-and-office/i-stock-my-toolkit-with-these-10-diy-gadgets-and-theyre-all-i-need/",
    "https://www.reddit.com/r/diyelectronics/comments/19djbtr/what_are_some_cool_gadgets_that_absolute/",
    "https://interestingengineering.com/lists/7-homemade-gadgets-to-inspire-the-inventor-within-you"
  ],
  "common_keywords": [
    "DIY",
    "gadgets",
    "tutorial",
    "beginner",
    "tools"
  ],
  "content_structure_insights": [
    "Most articles structure content as numbered or bullet-point lists with detailed gadget descriptions and sourcing information",
    "Step-by-step instructions and purchasing links are commonly included, often accompanied by images and external references"
  ],
  "key_themes_and_angles": [
    "Professional tool recommendations for home improvement (ZDNET) vs. beginner electronics projects (Reddit) vs. creative repurposing of household items (IE)",
    "Emphasis on low-cost, accessible materials and practical solutions for DIY enthusiasts"
  ],
  "executive_summary": "The competitive landscape shows diverse approaches from professional tool reviews to beginner tutorials and creative DIY projects. To stand out, create content that combines clear step-by-step guides with cost-effective, safety-conscious solutions, addressing specific user needs like accessibility for beginners and durable, multi-use gadgets."
}

================================================================================
"""

import os
import re

import asyncio
import json
import logging
from typing import List, Dict, Any

from src.function_calling import FunctionCalling
from src.function_calling.tools import WebScraper, URLContextTool
from src.function_calling.tools.web_search import WebSearchTool
from dotenv import load_dotenv
load_dotenv()

# --- Configuration ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Load your analyzer model
ANALYZER_MODEL = os.getenv("ANALYZER_MODEL")
# The number of top competitors to analyze.
TOP_N_COMPETITORS = 3
# Minimum security score for a URL to be considered credible for analysis.
MIN_SECURITY_SCORE = 60


async def run_competitor_analysis(topic: str):
    """
    Orchestrates the entire competitor analysis workflow.

    @param topic The keyword or topic to research (e.g., "best sustainable coffee beans 2025").
    @returns A dictionary containing the structured analysis report.
    @throws Exception if the process fails at any critical step.
    """
    logger.info(f"🚀 Starting competitor analysis for topic: '{topic}'")

    competitor_urls = []
    competitor_content = ""

    # Use async context managers for robust resource handling.
    async with WebSearchTool() as searcher, WebScraper() as scraper, URLContextTool() as url_context:

        # --- Step 1: Find Competitors using Web Search ---
        logger.info("Step 1: Searching for top competitors...")
        search_query = f"top ranking articles for '{topic}'"
        search_response = await searcher.search(search_query, num_results=TOP_N_COMPETITORS * 2, parse_result_for_llm=False)

        if not search_response or not search_response.results:
            logger.error("Failed to retrieve search results. Aborting.")
            raise Exception("Web search failed.")

        logger.info(
            f"Found {len(search_response.results)} potential competitors.")

        # --- Step 2: Validate and Scrape Competitor Content ---
        logger.info(
            f"Step 2: Validating and scraping content from the top {TOP_N_COMPETITORS} credible sources...")

        for result in search_response.results:
            if len(competitor_urls) >= TOP_N_COMPETITORS:
                break  # Stop once we have enough valid competitors.

            url = result.link
            if not url:
                continue

            # Validate URL security and credibility.
            context = url_context.analyze_url(url)
            security_score = context.get(
                'security', {}).get('score', 0)

            if security_score >= MIN_SECURITY_SCORE:
                logger.info(
                    f"  ✅ URL '{url}' is credible (Score: {security_score}). Scraping content...")
                try:
                    # Scrape the text content from the validated URL.
                    content = await scraper.scrape_text(url)
                    if content and "Could not retrieve" not in content:
                        competitor_urls.append(url)
                        # Limit content length
                        competitor_content += f"\n\n--- Content from {url} ---\n{content}"
                    else:
                        logger.warning(
                            f"  ⚠️ Could not scrape significant content from '{url}'.")
                except Exception as e:
                    logger.error(f"  ❌ Failed to scrape '{url}': {e}")
            else:
                logger.warning(
                    f"  Skipping URL '{url}' due to low security score ({security_score}).")

    if not competitor_content:
        logger.error(
            f"Could not gather content from any of {len(search_response.results)} search results. "
            f"Common causes: low security scores, scraping failures, or empty pages.")
        raise Exception("Content scraping failed for all competitors.")

    logger.info(
        f"Successfully scraped content from {len(competitor_urls)} competitors.")

    # --- Step 3: AI-Powered Analysis and Report Generation ---
    logger.info("Step 3: Generating final analysis report with AI...")
    analyzer_agent = FunctionCalling(
        model=ANALYZER_MODEL, max_turns=1, temperature=0.4)

    analysis_prompt = f"""
    As a world-class SEO strategist, analyze the provided content from top-ranking articles for the topic "{topic}". 
    Based *only* on the text below, generate a competitor analysis report.

    **COMPETITOR CONTENT:**
    {competitor_content}

    **REPORT REQUIREMENTS:**
    Your response MUST be a single JSON object. Do not include any text outside of the JSON.
    The JSON object must have the following structure:
    {{
        "topic_analyzed": "{topic}",
        "competitors_analyzed": ["url1", "url2", ...],
        "common_keywords": ["keyword1", "keyword2", "keyword3"],
        "content_structure_insights": [
            "Insight about common sections (e.g., 'Most articles include a 'How to Choose' section').",
            "Insight about formatting (e.g., 'Use of bullet points and numbered lists is prevalent')."
        ],
        "key_themes_and_angles": [
            "A dominant theme identified (e.g., 'Emphasis on ethical sourcing and certifications').",
            "A unique angle taken by competitors (e.g., 'Focus on health benefits and antioxidant properties')."
        ],
        "executive_summary": "A 2-3 sentence summary of the competitive landscape and recommendations for creating a superior piece of content."
    }}
    """

    report_str, _ = await analyzer_agent.run_async(analysis_prompt)

    # --- Step 4: Parse and Return the Final Report ---
    logger.info("Step 4: Finalizing the report.")
    try:
        # Robustly find the JSON object within the agent's response.
        json_match = re.search(r'\{[\s\S]*\}', report_str)
        if not json_match:
            raise json.JSONDecodeError(
                "No JSON object found in the response.", report_str, 0)

        report_json = json.loads(json_match.group(0))
        # Add the list of analyzed URLs to the final report
        report_json["competitors_analyzed"] = competitor_urls
        return report_json
    except (json.JSONDecodeError, AttributeError) as e:
        logger.error(
            f"Failed to parse AI-generated report: {e}\nRaw response: {report_str}")
        raise Exception("Failed to generate a valid analysis report from AI.")


async def main():
    """Main function to run the competitor analysis demo."""
    print("="*80)
    print("💡 SEO Agent Demo: Automated Competitor Research 💡")
    print("="*80)
    print("This script demonstrates how to analyze top-ranking competitors for a given topic to inform SEO strategy.")

    # The topic for which to perform the analysis.
    target_topic = "Home made gadgets"

    try:
        final_report = await run_competitor_analysis(target_topic)

        print("\n\n✅ Analysis Complete! Here is your report:\n")
        print(json.dumps(final_report, indent=2, ensure_ascii=False))

    except Exception as e:
        print(f"\n\n❌ An error occurred during the analysis: {e}")

    print("\n" + "="*80)


if __name__ == "__main__":
    # This allows the script to be run from the command line.
    # It requires the OPENROUTER_API_KEY environment variable to be set.
    import os
    import re
    if not os.getenv("OPENROUTER_API_KEY"):
        print("FATAL ERROR: OPENROUTER_API_KEY environment variable not set.")
    else:
        asyncio.run(main())
