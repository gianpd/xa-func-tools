import inspect
import json
from typing import Callable, List, Dict, Any

from src.function_calling.models import Tool, ToolParameters, ToolParameterProperty, Function

import inspect
from typing import Callable, Dict, Any, get_type_hints, Union, List, Optional
import re


def create_tool_from_function(func: Callable, exclude_params: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Creates a tool definition from a Python function in the exact template format.
    
    Args:
        func: The function to create the tool from.
        exclude_params: List of parameter names to exclude from the tool definition.
        
    Returns:
        A dictionary representing the tool in the template format.
        
    Raises:
        ValueError: If function lacks proper docstring or has unsupported types.
    """
    if exclude_params is None:
        exclude_params = ['df', 'self', 'cls']
    
    signature = inspect.signature(func)
    docstring = inspect.getdoc(func)
    
    if not docstring:
        raise ValueError(f"Function '{func.__name__}' must have a docstring.")
    
    print(signature)
    print(docstring)
    
    # Parse docstring for description and parameter info
    description, param_descriptions = _parse_docstring(docstring)
    
    # Get type hints for better type inference
    type_hints = get_type_hints(func)
    
    # Build properties and required lists
    properties = {}
    required = []
    
    for name, param in signature.parameters.items():
        if name in exclude_params:
            continue
            
        # Determine parameter type
        param_type = _get_parameter_type(param, type_hints.get(name))
        
        # Get parameter description
        param_desc = param_descriptions.get(name, f"Parameter {name}")
        
        # Build property definition
        prop_def = {
            "type": param_type["type"],
            "description": param_desc
        }
        
        # Add additional constraints if present
        if "enum" in param_type:
            prop_def["enum"] = param_type["enum"]
        if "format" in param_type:
            prop_def["format"] = param_type["format"]
        if "pattern" in param_type:
            prop_def["pattern"] = param_type["pattern"]
        if "minimum" in param_type:
            prop_def["minimum"] = param_type["minimum"]
        if "maximum" in param_type:
            prop_def["maximum"] = param_type["maximum"]
            
        properties[name] = prop_def
        
        # Check if parameter is required
        if param.default == inspect.Parameter.empty:
            required.append(name)
    
    # Build the tool definition in the exact template format
    # Build the tool definition in OpenAI function calling format
    tool_definition = {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": description.strip(),
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }
    }
    
    return tool_definition


def _parse_docstring(docstring: str) -> tuple[str, Dict[str, str]]:
    """
    Parse docstring to extract description and parameter descriptions.
    
    Supports multiple docstring formats:
    - Google style: Args: param_name: description
    - NumPy style: Parameters: param_name : type, description
    - Simple format: @param param_name: description
    - Basic format: param_name: description
    """
    lines = [line.strip() for line in docstring.strip().split('\n')]
    
    # Find the main description (everything before Args/Parameters/@param)
    description_lines = []
    param_section_start = None
    
    for i, line in enumerate(lines):
        if line.lower().startswith(('args:', 'arguments:', 'parameters:', 'param:')):
            param_section_start = i
            break
        elif line.startswith('@param'):
            param_section_start = i
            break
        elif ':' in line and i > 0:  # Simple param format
            param_section_start = i
            break
        else:
            description_lines.append(line)
    
    description = ' '.join(description_lines).strip()
    
    # Parse parameter descriptions
    param_descriptions = {}
    
    if param_section_start is not None:
        param_lines = lines[param_section_start:]
        
        for line in param_lines:
            # Google/NumPy style: "param_name: description" or "param_name : type, description"
            if ':' in line and not line.lower().startswith(('args:', 'arguments:', 'parameters:', 'returns:', 'return:')):
                if line.startswith('@param'):
                    # @param format: "@param param_name: description"
                    match = re.match(r'@param\s+(\w+):\s*(.+)', line)
                    if match:
                        param_name, param_desc = match.groups()
                        param_descriptions[param_name] = param_desc.strip()
                else:
                    # Standard format: "param_name: description" or "param_name : type, description"
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        param_name = parts[0].strip()
                        param_desc = parts[1].strip()
                        
                        # Handle NumPy style where description might start after type
                        if ',' in param_desc:
                            param_desc = param_desc.split(',', 1)[1].strip()
                        
                        param_descriptions[param_name] = param_desc
    
    return description, param_descriptions


def _get_parameter_type(param: inspect.Parameter, type_hint: Any = None) -> Dict[str, Any]:
    """
    Determine the JSON Schema type for a parameter based on its annotation and type hint.
    """
    param_type = {"type": "string"}  # Default fallback
    
    # Check type hint first, then annotation
    type_to_check = type_hint if type_hint is not None else param.annotation
    
    if type_to_check == inspect.Parameter.empty:
        return param_type
    
    # Handle basic types
    if type_to_check == int:
        param_type = {"type": "integer"}
    elif type_to_check == float:
        param_type = {"type": "number"}
    elif type_to_check == bool:
        param_type = {"type": "boolean"}
    elif type_to_check == str:
        param_type = {"type": "string"}
    elif type_to_check == list:
        param_type = {"type": "array"}
    elif type_to_check == dict:
        param_type = {"type": "object"}
    
    # Handle Union types (like Optional)
    elif hasattr(type_to_check, '__origin__'):
        if type_to_check.__origin__ is Union:
            # Handle Optional[Type] -> Union[Type, None]
            args = type_to_check.__args__
            non_none_types = [arg for arg in args if arg is not type(None)]
            if len(non_none_types) == 1:
                return _get_parameter_type(param, non_none_types[0])
            else:
                # Multiple non-None types, default to string
                param_type = {"type": "string"}
        elif type_to_check.__origin__ is list:
            param_type = {"type": "array"}
            if type_to_check.__args__:
                items_type = _get_parameter_type(param, type_to_check.__args__[0])
                param_type["items"] = items_type
        elif type_to_check.__origin__ is dict:
            param_type = {"type": "object"}
    
    # Handle string enums (if parameter has specific values in docstring)
    if param_type["type"] == "string" and hasattr(param, 'default') and param.default != inspect.Parameter.empty:
        if isinstance(param.default, (list, tuple)):
            param_type["enum"] = list(param.default)
    
    # Add format hints for common string patterns
    if param_type["type"] == "string":
        param_name_lower = param.name.lower()
        if any(keyword in param_name_lower for keyword in ['email', 'mail']):
            param_type["format"] = "email"
        elif any(keyword in param_name_lower for keyword in ['url', 'uri', 'link']):
            param_type["format"] = "uri"
        elif any(keyword in param_name_lower for keyword in ['date']):
            param_type["format"] = "date"
        elif any(keyword in param_name_lower for keyword in ['time']):
            param_type["format"] = "time"
        elif any(keyword in param_name_lower for keyword in ['path', 'file']):
            param_type["description"] = param_type.get("description", "") + " (file path)"
    
    return param_type

if __name__ == "__main__":

    from function_calling.tools.url_context import URLContextTool
    from function_calling.tools.premium_web_search import SerperLikeWebSearcher
    from function_calling.tools.web_scraper import WebScraper

    url_context = URLContextTool()
    search = SerperLikeWebSearcher()
    scraper = WebScraper()

    ffs = {
        f.__name__: create_tool_from_function(f) for f in [url_context.get_full_context, search.search, scraper.scrape_structured_data]
    }

    print(ffs)