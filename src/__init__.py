"""
Top-level module for the AI function calling and reasoning package.

This module provides access to core functionality for function calling and reasoning capabilities.
It includes key classes and functions from the `function_calling` and `reasoning` subpackages.
"""

# Explicit imports from function_calling subpackage
from .function_calling.core import FunctionCalling
from .function_calling.models import Field, Function, Message, BaseModel
from .function_calling.utils.utils import create_tool_from_function

# Define public API for `from module import *`
__all__ = [
    "FunctionCalling",
    "Field",
    "Function",
    "Message",
    "BaseModel",
    "create_tool_from_function",
    "call_llm"
]

__version__ = "1.0.0"
__author__ = "gianpd"
__license__ = "MIT"

# Note: This package requires Python 3.8+ and has dependencies on openai, pydantic, and asyncio-compatible libraries.
# For full functionality, install: pip install openai pydantic asyncio