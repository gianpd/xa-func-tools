# xa-func-tools: Tool-Augmented AI Agent Framework

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/gianpd/xa-func-tools/blob/main/LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/gianpd/xa-func-tools?style=social)](https://github.com/gianpd/xa-func-tools)
[![GitHub Forks](https://img.shields.io/github/forks/gianpd/xa-func-tools?style=social)](https://github.com/gianpd/xa-func-tools/fork)

**xa-func-tools** is an open-source Python framework designed to build intelligent, tool-augmented AI agents, by prioritize AI explainability concepts. It empowers developers to create autonomous systems that leverage large language models (LLMs) for reasoning, planning, and executing tasks via function calling. Built with a focus on web interactions, document processing, and agentic workflows, this framework bridges the gap between LLMs and real-world tools, enabling applications like automated research, web scraping agents, and secure data extraction.

Whether you're prototyping a simple web browser agent or scaling to multi-tool research pipelines, xa-func-tools provides a modular, async-friendly foundation. It's inspired by agentic patterns like ReAct (Reasoning and Acting) and draws from libraries such as LangChain and AutoGen, but emphasizes simplicity, security (e.g., mandatory URL checks), and extensibility.

## Why xa-func-tools?
In an era where AI agents are transforming automation, xa-func-tools stands out by:
- Prioritizing **safety and explainability**: Agents must reason step-by-step and validate URLs before interactions.
- Supporting **hybrid content handling**: Seamlessly process web pages, PDFs, and documents with built-in truncation management.
- Complete XA logs: Every call logs metadata (timestamps, hashes, token estimates, error types), thoughts, and even generates analysis reports with stats on   durations, errors, and tool usage. This supports "post-hoc" explainability: After a run, you can reconstruct exactly what happened.

- Encouraging **open collaboration**: As a truly open-source project (MIT licensed), we invite contributions to evolve it into a community-driven ecosystem.

This framework is ideal for developers, researchers, and hobbyists building AI-powered tools for data gathering, analysis, or automation without the overhead of larger frameworks (at the end of the game is just an api call, isn't?).

## Key Features
- **Agentic Loop with Function Calling**: Multi-turn interactions where the LLM reasons, plans, and calls tools dynamically via OpenRouter/OpenAI-compatible APIs.
- **Built-in Tools**:
  - Web Scraping & Browsing: Extract text, links, structured data; interactive navigation with Playwright.
  - Search: Web and X (Twitter) searches with snippets, semantic filtering, and advanced operators.
  - Document Extraction: Handle PDFs/DOCX (local/remote) using MarkItDown for text processing.
  - URL Analysis: Security scoring, connectivity checks, and metadata extraction before any interaction.
  - ArXiv Integration: Search and retrieve papers by query, author, or category.
- **Async & Robust Design**: Context managers for tool lifecycles, rate limiting, retries, and proxy support.
- **Explainability & Logging**: Agent "thoughts" before actions; comprehensive logging with metadata and analysis reports.
- **Customization**: Easy tool registration; configurable system prompts, temperatures, and max turns.
- **Examples Included**: Ready-to-run demos for web scraping agents, premium research agents, and more.

## How It Works
At its core, xa-func-tools revolves around the `FunctionCalling` class:
1. **Initialization**: Set up with an LLM model (e.g., via OpenRouter), system prompt, and max turns for safety.
2. **Tool Registration**: Add functions as tools—automatically converts signatures to JSON schemas.
3. **Agentic Execution**: Run an async loop where the LLM:
   - Receives a user prompt.
   - Outputs a "thought" (reasoning step).
   - Calls tools if needed (e.g., scrape a URL after security check).
   - Processes tool results (condensed for token efficiency).
   - Repeats until resolved or max turns reached.
4. **Output**: Final answer plus execution log for debugging.

This ReAct-inspired flow ensures agents are deliberate and traceable. For instance, in a research agent:
- Generate queries.
- Search and validate sources.
- Extract/synthesize data.
- Output structured reports.

The framework handles async operations natively, making it suitable for I/O-bound tasks like web interactions.

## Installation
```bash
git clone https://github.com/gianpd/xa-func-tools.git
cd xa-func-tools
pip install -e .  # Install core deps (openai, aiohttp, playwright, etc.)
playwright install  # For browser tools
```

Set environment variables:
- `OPENROUTER_API_KEY`: For LLM access.
- Optional: `SERPAPI_KEY` for web search.

## Quick Start
```python
import asyncio
from src.function_calling import FunctionCalling
from src.function_calling.tools import WebScraper

async def main():
    async with WebScraper() as scraper:
        agent = FunctionCalling(model="your-model", max_turns=5)
        agent.register_tool(scraper.scrape_text)
        
        prompt = "Extract text from https://example.com"
        answer, log = await agent.run_async(prompt)
        print(answer)

asyncio.run(main())
```

See `examples/` for advanced demos like premium research agents.

## What It Can Do
- **Automated Research**: Synthesize data from web/PDF sources with credibility scoring (e.g., `web_premium_scraper.py`).
- **Web Automation**: Navigate, interact, and screenshot pages securely.
- **Data Extraction**: Batch process documents, scrape structured data, or search ArXiv.
- **Custom Agents**: Build specialized agents for tasks like event monitoring or content analysis.

Limitations: Relies on external APIs; some tools (e.g., PDFs) may require additional libs. Files can be truncated in long outputs—use tools to browse full content.

## Roadmap & Opportunities for Improvement
While robust, xa-func-tools is evolving:
- **Memory & State**: Add persistent memory (e.g., vector DB integration).
- **Multi-Agent Support**: Enable collaboration between agents.
- **UI/Deployment**: Add Streamlit/Flask wrappers for web apps.
- **Testing & Benchmarks**: Expand unit tests and compare with SOTA.

We believe in collaborative growth—join us to make it better!

## Contributing
xa-func-tools thrives on community input! Whether fixing bugs, adding tools, or suggesting features, your contributions are welcome. Fork the repo, create a branch, and submit a PR. Check issues for open tasks.

- **Guidelines**: Follow PEP 8; add tests/docs for new features.
- **Ideas?**: Open an issue to discuss enhancements like ethical AI integrations.
- **License**: MIT—use, modify, and share freely.

Together, let's build the next generation of secure, explainable AI agents!

## License
MIT © [gianpd](https://github.com/gianpd) and contributors.