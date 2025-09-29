"""
/**
* @file web_browser.py
* @purpose An agentic web browser tool using Playwright for web interaction and screenshot capture.
*
* @dependencies
* - playwright: For browser automation.
* - aiofiles: For async file operations.
* - tempfile: For secure temporary file handling.
*
* @notes
* - Now supports configurable headless mode for easier debugging.
* - Sets a default User-Agent to avoid basic bot detection.
* - Manages browser lifecycle via async context manager.
* - Added select_option and find_and_click_by_text for advanced form interaction.
* - Enhanced with reliable screenshot capture from HTML content.
*/
"""

import asyncio
import time
import os
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any
from playwright.async_api import async_playwright, Page, Browser

class WebBrowser:
    """
    An agentic web browser tool that can navigate, read, interact with web pages,
    and capture screenshots from HTML content.
    It maintains a single browser instance for the duration of its lifecycle.
    """

    def __init__(self, headless: bool = True):
        """
        Initializes the WebBrowser instance.

        @param headless Whether to run the browser in headless mode. Defaults to True.
        """
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None
        self._playwright = None
        self.headless = headless

    async def __aenter__(self):
        """
        Asynchronous context manager entry to launch the browser.
        This allows for graceful startup and shutdown.
        """
        self._playwright = await async_playwright().start()
        self.browser = await self._playwright.chromium.launch(headless=self.headless)
        self.page = await self.browser.new_page()
        await self.page.set_extra_http_headers({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Asynchronous context manager exit to close the browser.
        """
        if self.page:
            await self.page.close()
        if self.browser:
            await self.browser.close()
        if self._playwright:
            await self._playwright.stop()

    async def navigate(self, url: str) -> str:
        """
        Navigates to a specified URL.
        
        @param url The URL to navigate to.
        @returns A string indicating success or failure.
        """
        if not self.page:
            return "Browser not initialized. Please start the browser session."
        try:
            await self.page.goto(url, wait_until='domcontentloaded', timeout=60000)
            return f"Successfully navigated to {url}. Current page title: '{await self.page.title()}'"
        except Exception as e:
            return f"Error navigating to {url}: {e}"

    async def get_text_content(self) -> str:
        """
        Retrieves the visible text content of the current page.

        @returns The text content of the page, or an error message.
        """
        if not self.page:
            return "Browser not initialized."
        try:
            return await self.page.inner_text('body')
        except Exception as e:
            return f"Error getting text content: {e}"

    async def find_and_click(self, selector: str) -> str:
        """
        Finds an element using a CSS selector and clicks it.

        @param selector The CSS selector of the element to click.
        @returns A string indicating success or failure.
        """
        if not self.page:
            return "Browser not initialized."
        try:
            await self.page.click(selector, timeout=10000)
            return f"Clicked on element with selector: '{selector}'."
        except Exception as e:
            return f"Error finding or clicking element with selector '{selector}': {e}"

    async def find_and_click_by_text(self, text: str) -> str:
        """
        Finds an element by its exact text content and clicks it. Very useful for buttons.

        @param text The exact text of the element to click (case-sensitive).
        @returns A string indicating success or failure.
        """
        if not self.page:
            return "Browser not initialized."
        try:
            # Use Playwright's powerful text selector
            await self.page.click(f'text="{text}"', timeout=10000)
            return f"Clicked on element with text: '{text}'."
        except Exception as e:
            return f"Error finding or clicking element with text '{text}': {e}"

    async def find_and_type(self, selector: str, text: str) -> str:
        """
        Finds an input element and types text into it.

        @param selector The CSS selector of the input element.
        @param text The text to type into the element.
        @returns A string indicating success or failure.
        """
        if not self.page:
            return "Browser not initialized."
        try:
            await self.page.fill(selector, text, timeout=10000)
            return f"Typed '{text}' into element with selector '{selector}'."
        except Exception as e:
            return f"Error finding or typing into element with selector '{selector}': {e}"

    async def select_option(self, selector: str, value: str) -> str:
        """
        Selects an option from a dropdown (<select> element).

        @param selector The CSS selector of the <select> element.
        @param value The 'value' attribute of the <option> to select.
        @returns A string indicating success or failure.
        """
        if not self.page:
            return "Browser not initialized."
        try:
            await self.page.select_option(selector, value=value)
            return f"Selected option with value '{value}' from element '{selector}'."
        except Exception as e:
            return f"Error selecting option '{value}' from element with selector '{selector}': {e}"

    async def save_screenshot_from_url(
            self,
            url: str,
            output_dir: str = "cards",
            wait_time: float = 1.0,
            viewport_size: Dict[str, int] = None,
            filename_prefix: str = "screenshot",
            timeout: float = 30.0
        ) -> str:
        """
        Navigates to a URL and saves a screenshot as PNG file.
        
        @param url The URL to navigate to and capture.
        @param output_dir The directory to save the screenshot. Defaults to "cards".
        @param wait_time Time in seconds to wait after loading page for CSS/animations. Defaults to 1.0.
        @param viewport_size Dictionary with 'width' and 'height' for viewport size. Defaults to 1024x1024.
        @param filename_prefix Prefix for the output filename. Defaults to "screenshot".
        @param timeout Maximum time in seconds to wait for page load. Defaults to 30.0.
        @returns Path to the saved screenshot file.
        @raises RuntimeError If browser is not initialized or screenshot fails.
        @raises ValueError If URL is invalid or empty.
        """
        if not self.browser:
            raise RuntimeError("Browser not initialized. Please start the browser session first.")
        
        if not url or not url.strip():
            raise ValueError("URL cannot be empty or None.")
        
        # Add protocol if missing
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        if viewport_size is None:
            viewport_size = {"width": 1024, "height": 1024}
        
        page = None
        temp_file = None
        
        try:
            # Create a new page to avoid interfering with current state
            page = await self.browser.new_page()
            await page.set_viewport_size(viewport_size)
            
            # Navigate to the URL with timeout
            await page.goto(url, timeout=timeout * 1000)  # Playwright expects milliseconds
            
            # Wait for the page to fully load and render
            await page.wait_for_load_state('networkidle', timeout=timeout * 1000)
            
            # Additional wait for CSS styles and animations to load
            await asyncio.sleep(wait_time)
            
            # Create a temporary file for the screenshot
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            temp_file_path = temp_file.name
            temp_file.close()  # Close the file so Playwright can write to it
            
            # Take screenshot
            await page.screenshot(path=temp_file_path, full_page=True)
            
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate unique filename with timestamp
            timestamp = int(time.time())
            filename = f"{filename_prefix}_{timestamp}.png"
            output_path = os.path.join(output_dir, filename)
            
            # Move the temporary file to the final destination
            os.rename(temp_file_path, output_path)
            
            return output_path
            
        except Exception as e:
            # Clean up temporary file if it exists
            if temp_file and os.path.exists(temp_file.name):
                try:
                    os.unlink(temp_file.name)
                except OSError:
                    pass
            raise RuntimeError(f"Failed to save screenshot from URL '{url}': {str(e)}")
            
        finally:
            # Close the page if it was created
            if page:
                await page.close()