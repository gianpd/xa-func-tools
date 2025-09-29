# examples/web_browsing_agent.py

import asyncio
from src.function_calling import FunctionCalling
from src.function_calling.tools import WebBrowser, WebScraper

async def main():
    """
    An example of using the WebBrowser agent to search for information.
    """
    async with WebBrowser() as browser, WebScraper() as scrape:
        handler = FunctionCalling(model="z-ai/glm-4.5", max_turns=15)

        # Register the browser's methods as tools
        handler.register_tool(browser.navigate)
        handler.register_tool(browser.get_text_content)
        handler.register_tool(browser.find_and_click)
        handler.register_tool(browser.find_and_type)
        handler.register_tool(scrape.scrape_structured_data)
        handler.register_tool(browser.save_screenshot_from_url)

        user_prompt = (
            "Please navigate to b2bdeals.store "
            "and take a screenshot from there"
        )

        final_answer = await handler.run_async(
            user_prompt,
            tool_choice="auto",
            external_tools={"browser": browser}
        )
        
        print("\n--- Final Answer ---")
        print(final_answer)

if __name__ == "__main__":
    # On Windows, you might need to set a different event loop policy for Playwright
    # asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())