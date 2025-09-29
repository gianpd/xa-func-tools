# xa-func-tools

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/gianpd/xa-func-tools/blob/main/LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/gianpd/xa-func-tools?style=social)](https://github.com/gianpd/xa-func-tools)
[![GitHub Forks](https://img.shields.io/github/forks/gianpd/xa-func-tools?style=social)](https://github.com/gianpd/xa-func-tools/fork)


A lightweight, transparent Python framework for building AI agents with function calling. Designed to make agent behavior debuggable and understandable.

## What is this?

**xa-func-tools** helps you build AI agents that can use tools—scrape websites, search the web, process documents, and more. It wraps OpenAI-compatible APIs with a focus on making agent decisions visible and traceable.

If you've worked with LangChain or AutoGen and wanted something simpler to understand and modify, this might be for you.

## Why use it?

**Transparency first:** Every agent action is logged with reasoning steps, metadata, and execution traces. When something goes wrong (and it will), you can see exactly what happened.

**Security by default:** URL validation is mandatory before any web interaction. No accidental requests to random URLs.

**Async-native:** Built for I/O-bound tasks like web scraping and API calls. No blocking operations holding up your agent loops.

**Easy to extend:** Register any Python function as a tool. The framework handles JSON schema generation and function calling automatically.

**Honest about limitations:** This isn't solving hallucinations or replacing production frameworks. It's a learning tool and prototyping environment that prioritizes clarity over features.

## Core Features

- **ReAct-style agent loop:** Agents think out loud before acting, making reasoning visible
- **Built-in tools:** Web scraping (Playwright), document extraction (PDF/DOCX), ArXiv search, URL analysis, web/X search
- **Comprehensive logging:** Timestamps, token estimates, error tracking, and post-run analysis reports
- **Async context managers:** Clean resource handling for browser sessions and HTTP clients
- **Rate limiting & retries:** Production-ready error handling for flaky APIs
- **Configurable system prompts:** Customize agent behavior without touching core code

## Quick Start

```bash
git clone https://github.com/gianpd/xa-func-tools.git
cd xa-func-tools
pip install -e .
playwright install  # Only if using browser tools
```

Set your API key:
```bash
export OPENROUTER_API_KEY="your-key-here"
# Optional: export SERPAPI_KEY="your-key" for web search
```

Basic example:
```python
import asyncio
from src.function_calling import FunctionCalling
from src.function_calling.tools import WebScraper

async def main():
    async with WebScraper() as scraper:
        agent = FunctionCalling(model="qwen/qwen3-next-80b-a3b-thinking", max_turns=5)
        agent.register_tool(scraper.scrape_text)
        
        answer, log = await agent.run_async("What's on the front page of example.com?")
        print(answer)
        print(f"\nAgent took {len(log)} turns")

asyncio.run(main())
```

Check the `examples/` folder for more complex demos like multi-source research agents.

## How It Works

1. You create a `FunctionCalling` instance with an LLM model and system prompt
2. Register tools (any Python function) using `register_tool()`
3. Call `run_async()` with your prompt
4. The agent loops: think → call tools → process results → repeat
5. Returns final answer + full execution log

Each loop iteration logs:
- Agent's reasoning ("thought")
- Which tool was called and with what arguments
- Tool results (truncated if needed for token efficiency)
- Errors, durations, token estimates

The log is structured JSON you can analyze programmatically.

## What You Can Build

- **Research assistants:** Gather and synthesize information from multiple sources
- **Web automation:** Navigate sites, extract data, generate reports
- **Document processors:** Batch analyze PDFs, papers, or articles
- **Monitoring bots:** Track websites or feeds for specific content
- **Custom agents:** Combine tools to solve your specific problems

## Limitations (The Honest Part)

- **No memory persistence yet:** State is lost between runs. Vector DB integration is on the roadmap.
- **Single agent only:** No multi-agent orchestration or collaboration patterns.
- **Basic error recovery:** Retries and logging, but no sophisticated failure strategies.
- **Token management is manual:** You need to watch context windows yourself.
- **Not production-tested at scale:** This is a learning/prototyping tool, not enterprise software.

## Roadmap

Planned improvements:
- Persistent memory with vector database support
- Multi-agent coordination patterns
- Web UI for monitoring and debugging
- Benchmark suite comparing performance to other frameworks
- More sophisticated error recovery strategies

PRs and ideas welcome—see Contributing below.

## Contributing

This project thrives on community input. Whether you're fixing a bug, adding a tool, or improving documentation, contributions are appreciated.

**How to contribute:**
1. Fork the repo
2. Create a feature branch
3. Make your changes (follow PEP 8)
4. Add tests if applicable
5. Submit a PR with clear description

**Need ideas?** Check the Issues tab for open tasks or suggest your own improvements.

## Philosophy

At the end of the day, agent frameworks are thin wrappers around LLM API calls. The value isn't in complexity—it's in making those calls transparent, safe, and easy to debug. That's what xa-func-tools tries to do.

If you want a production framework with every feature, use LangChain. If you want to understand how agents work and build something custom, start here.

## License

MIT © [gianpd](https://github.com/gianpd) and contributors.

Use it, modify it, learn from it. That's what open source is for.