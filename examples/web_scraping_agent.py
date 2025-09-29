# examples/web_scraping_agent.py

import asyncio
import json
from src.function_calling import FunctionCalling
from src.function_calling.tools import WebScraper, WebSearchTool, WebBrowser, URLContextTool # This import still works thanks to the wrapper

async def main():
    """
    An agentic example that uses the new PremiumWebSearcher to find URLs
    and the WebScraper to extract information from those pages.
    """
    # The agent will now use two tools that should be managed as context managers
    scraper = WebScraper()
    searcher = WebSearchTool()
    url_context = URLContextTool()
    browser = WebBrowser()
    
    handler = FunctionCalling(
        model="z-ai/glm-4.5",
        max_turns=10
    )

    # CHANGED: We now use 'async with' for both the scraper and the new searcher
    # This ensures their resources (like network sessions) are handled correctly.
    async with scraper, searcher, browser, url_context:
        try:
            # Register the high-level tools for the agent
            handler.register_tool(url_context.analyze_url)
            handler.register_tool(searcher.search)
            handler.register_tool(browser.find_and_click)
            handler.register_tool(browser.get_text_content)
            handler.register_tool(scraper.scrape_text)
            handler.register_tool(scraper.scrape_with_selector)
            handler.register_tool(scraper.scrape_links)
            handler.register_tool(scraper.scrape_structured_data)
            
            user_prompt = (
               "Find the official website of openai"
            )

            final_answer, execution_log = await handler.run_async(
                user_prompt,
                tool_choice="auto",
                # Pass the instantiated tools to the handler
                external_tools={"searcher": searcher, "scraper": scraper, "url_conmtext": url_context}
            )
            
            print("\n" + "="*50)
            print("✅ --- Agent's Final Answer ---")
            print(final_answer)
            print("="*50 + "\n")

            print("🔎 --- Full Execution Log ---")
            print(json.dumps(execution_log, indent=2))
            print("="*50)

        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())