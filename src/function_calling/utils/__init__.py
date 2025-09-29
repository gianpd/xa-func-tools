"""
Utilities submodule for the function calling framework.

This module provides helper functions for tool creation, logging, and debugging.
It includes utilities for converting functions to tools and comprehensive LLM logging.

@dependencies
- utils.py: Core utility functions (standard library + typing)
- xa_logger.py: Advanced logging (standard library + json, threading)

@notes
- create_tool_from_function: Automatically generates OpenAI-compatible tool schemas from Python functions.
- LLMLogger: Non-intrusive logging wrapper; does not modify LLM call behavior.
- Error handling: Utilities raise ValueError for invalid inputs; loggers handle exceptions gracefully.
- Production-ready: Thread-safe, async-compatible, and low-overhead.
"""

from .utils import create_tool_from_function
from .xa_logger import LLMLogger, enable_llm_logging

# Define public API for import *
__all__ = [
    "create_tool_from_function",
    "LLMLogger", 
    "enable_llm_logging"
]

__version__ = "1.0.0"
__author__ = "gianpd"
__license__ = "MIT"