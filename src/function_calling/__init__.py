"""
Core module for the function calling framework.

This package provides the FunctionCalling class and supporting utilities for integrating
LLM-based function calling into applications. It supports OpenRouter API integration,
tool registration, and asynchronous execution.

@dependencies
- openai: For API client (pip install openai)
- pydantic: For data models and validation (pip install pydantic)
- asyncio: For async operations (standard library)

@notes
- All operations are async-safe. Use asyncio.run() for synchronous execution.
- Error handling: TypeError for invalid tools, APIError for OpenRouter issues.
- Security: API keys are loaded from environment variables only.
"""

# Core public API
from .core import FunctionCalling

# Data models
from .models import (
    Tool, 
    ToolParameters, 
    ToolParameterProperty, 
    Function, 
    Message, 
    BaseModel as FunctionBaseModel
)

# Utilities
from .utils.utils import create_tool_from_function

# Pre-configured tools (if any)
from .tools import (
    WebScraper, 
    URLContextTool, 
    ArxivTool,
    WebSearchTool,
    WebBrowser

)

# Configure __all__ for import *
__all__ = [
    "FunctionCalling",
    "Tool", 
    "ToolParameters", 
    "ToolParameterProperty", 
    "Function", 
    "Message", 
    "FunctionBaseModel",
    "WebSearchTool",
    "create_tool_from_function",
    "WebScraper",
    "WebBrowser",
    "URLContextTool", 
    "ArxivTool"
]

__version__ = "1.0.0"
__author__ = "gianpd"
__license__ = "MIT"