from pydantic import BaseModel, Field
from typing import List, Dict, Any, Callable

class ToolParameterProperty(BaseModel):
    """
    Defines a single property within the parameters of a tool.
    """
    type: str = Field(..., description="The data type of the parameter (e.g., 'string', 'integer').")
    description: str = Field(..., description="A description of the parameter.")

class ToolParameters(BaseModel):
    """
    Defines the parameters for a tool.
    """
    type: str = "object"
    properties: Dict[str, ToolParameterProperty]
    required: List[str]

class Function(BaseModel):
    """
    Represents the function within a tool, including its name, description, and parameters.
    """
    name: str = Field(..., description="The name of the function to be called.")
    description: str = Field(..., description="A description of what the function does.")
    parameters: ToolParameters

class Tool(BaseModel):
    """
    Represents a tool that the language model can use.
    """
    type: str = "function"
    function: Function

class Message(BaseModel):
    """
    Represents a message in the conversation history.
    """
    role: str
    content: str
    tool_calls: List[Dict[str, Any]] = None