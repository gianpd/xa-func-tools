import inspect
import json
from typing import Callable, List, Dict, Any, get_type_hints, Union, Optional
import re


def create_tool_from_function(
    func: Callable, 
    exclude_params: Optional[List[str]] = None,
    name_prefix: Optional[str] = None,
    strict_validation: bool = True
) -> Dict[str, Any]:
    """
    Creates a tool definition from a Python function compatible with all major LLM providers.
    
    Args:
        func: The function to create the tool from.
        exclude_params: List of parameter names to exclude from the tool definition.
        name_prefix: Optional prefix to add to function name to avoid duplicates.
        strict_validation: If True, enforces strict validation rules (OpenAI/Gemini compatible).
        
    Returns:
        A dictionary representing the tool in the template format.
        
    Raises:
        ValueError: If function lacks proper docstring or has validation errors.
    """
    if exclude_params is None:
        exclude_params = ['df', 'self', 'cls']
    
    signature = inspect.signature(func)
    docstring = inspect.getdoc(func)
    
    if not docstring:
        raise ValueError(f"Function '{func.__name__}' must have a docstring.")
    
    # Parse docstring for description and parameter info
    description, param_descriptions = _parse_docstring(docstring)
    
    if not description:
        raise ValueError(f"Function '{func.__name__}' must have a description in its docstring.")
    
    # Get type hints for better type inference
    try:
        type_hints = get_type_hints(func)
    except Exception as e:
        print(f"Warning: Could not get type hints for {func.__name__}: {e}")
        type_hints = {}
    
    # Build properties and required lists
    properties = {}
    required = []
    
    for name, param in signature.parameters.items():
        if name in exclude_params:
            continue
            
        # Determine parameter type
        param_type = _get_parameter_type(param, type_hints.get(name), strict_validation)
        
        # Get parameter description
        param_desc = param_descriptions.get(name, f"The {name} parameter")
        
        # Ensure description is not empty (Gemini requirement)
        if not param_desc or param_desc.strip() == "":
            param_desc = f"The {name} parameter"
        
        # Build property definition
        prop_def = {
            "type": param_type["type"],
            "description": param_desc.strip()
        }
        
        # Add additional constraints if present
        if "items" in param_type:
            prop_def["items"] = param_type["items"]
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
        if "additionalProperties" in param_type:
            prop_def["additionalProperties"] = param_type["additionalProperties"]
            
        properties[name] = prop_def
        
        # Check if parameter is required
        if param.default == inspect.Parameter.empty:
            required.append(name)
    
    # Validate that all required params exist in properties
    if strict_validation:
        for req_param in required:
            if req_param not in properties:
                raise ValueError(f"Required parameter '{req_param}' not found in properties")
    
    # Build function name with optional prefix
    function_name = func.__name__
    if name_prefix:
        function_name = f"{name_prefix}{function_name}"
    
    # Validate function name (alphanumeric, underscores, hyphens only)
    if strict_validation and not re.match(r'^[a-zA-Z0-9_-]+$', function_name):
        raise ValueError(f"Function name '{function_name}' contains invalid characters")
    
    # Build the tool definition in OpenAI function calling format
    tool_definition = {
        "type": "function",
        "function": {
            "name": function_name,
            "description": description.strip(),
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False  # Gemini/OpenAI requirement
            }
        }
    }
    
    return tool_definition


def validate_tools(tools: List[Dict[str, Any]], provider: str = "openai") -> tuple[bool, List[str]]:
    """
    Validate tool definitions for a specific provider.
    
    Args:
        tools: List of tool definitions
        provider: Target provider ("openai", "gemini", "anthropic")
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    seen_names = set()
    
    for i, tool in enumerate(tools):
        if "function" not in tool:
            errors.append(f"Tool {i}: Missing 'function' key")
            continue
        
        func = tool["function"]
        
        # Check for required fields
        if "name" not in func:
            errors.append(f"Tool {i}: Missing 'name'")
        else:
            # Check for duplicate names
            if func["name"] in seen_names:
                errors.append(f"Tool {i}: Duplicate function name '{func['name']}'")
            seen_names.add(func["name"])
            
            # Validate name format
            if not re.match(r'^[a-zA-Z0-9_-]+$', func["name"]):
                errors.append(f"Tool {i}: Invalid function name '{func['name']}'")
        
        if "description" not in func or not func["description"].strip():
            errors.append(f"Tool {i}: Missing or empty 'description'")
        
        if "parameters" not in func:
            errors.append(f"Tool {i}: Missing 'parameters'")
            continue
        
        params = func["parameters"]
        
        # Validate parameters structure
        if "type" not in params or params["type"] != "object":
            errors.append(f"Tool {i}: parameters.type must be 'object'")
        
        if "properties" not in params:
            errors.append(f"Tool {i}: Missing 'properties' in parameters")
        else:
            # Validate each property
            for prop_name, prop_def in params["properties"].items():
                if "type" not in prop_def:
                    errors.append(f"Tool {i}, property '{prop_name}': Missing 'type'")
                
                # OpenAI/Gemini: arrays must have items
                if prop_def.get("type") == "array" and "items" not in prop_def:
                    errors.append(f"Tool {i}, property '{prop_name}': array type missing 'items'")
                
                # Gemini: descriptions should not be empty
                if provider == "gemini":
                    if "description" not in prop_def or not prop_def["description"].strip():
                        errors.append(f"Tool {i}, property '{prop_name}': Missing or empty description")
        
        # Validate required fields exist in properties
        if "required" in params:
            props = params.get("properties", {})
            for req_param in params["required"]:
                if req_param not in props:
                    errors.append(f"Tool {i}: Required parameter '{req_param}' not in properties")
    
    return len(errors) == 0, errors


def _parse_docstring(docstring: str) -> tuple[str, Dict[str, str]]:
    """
    Parse docstring to extract description and parameter descriptions.
    """
    lines = [line.strip() for line in docstring.strip().split('\n')]
    
    description_lines = []
    param_section_start = None
    
    for i, line in enumerate(lines):
        if line.lower().startswith(('args:', 'arguments:', 'parameters:', 'param:')):
            param_section_start = i
            break
        elif line.startswith('@param'):
            param_section_start = i
            break
        elif ':' in line and i > 0:
            param_section_start = i
            break
        else:
            description_lines.append(line)
    
    description = ' '.join(description_lines).strip()
    
    param_descriptions = {}
    
    if param_section_start is not None:
        param_lines = lines[param_section_start:]
        
        for line in param_lines:
            if ':' in line and not line.lower().startswith(('args:', 'arguments:', 'parameters:', 'returns:', 'return:')):
                if line.startswith('@param'):
                    match = re.match(r'@param\s+(\w+):\s*(.+)', line)
                    if match:
                        param_name, param_desc = match.groups()
                        param_descriptions[param_name] = param_desc.strip()
                else:
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        param_name = parts[0].strip()
                        param_desc = parts[1].strip()
                        
                        if ',' in param_desc:
                            param_desc = param_desc.split(',', 1)[1].strip()
                        
                        param_descriptions[param_name] = param_desc
    
    return description, param_descriptions


def _get_parameter_type(param: inspect.Parameter, type_hint: Any = None, 
                       strict: bool = True) -> Dict[str, Any]:
    """
    Determine the JSON Schema type for a parameter.
    With strict=True, ensures full OpenAI/Gemini compatibility.
    """
    param_type = {"type": "string"}  # Safe default
    
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
        # CRITICAL: Always include items for arrays
        param_type = {"type": "array", "items": {"type": "string"}}
    elif type_to_check == dict:
        # For strict mode, specify additionalProperties
        param_type = {"type": "object"}
        if strict:
            param_type["additionalProperties"] = True
    
    # Handle Union types (like Optional)
    elif hasattr(type_to_check, '__origin__'):
        if type_to_check.__origin__ is Union:
            args = type_to_check.__args__
            non_none_types = [arg for arg in args if arg is not type(None)]
            if len(non_none_types) == 1:
                return _get_parameter_type(param, non_none_types[0], strict)
            else:
                param_type = {"type": "string"}
        elif type_to_check.__origin__ is list:
            param_type = {"type": "array"}
            if type_to_check.__args__:
                items_type = _get_parameter_type(param, type_to_check.__args__[0], strict)
                param_type["items"] = items_type
            else:
                # CRITICAL: Must have items
                param_type["items"] = {"type": "string"}
        elif type_to_check.__origin__ is dict:
            param_type = {"type": "object"}
            if strict:
                param_type["additionalProperties"] = True
    
    return param_type


if __name__ == "__main__":
    # Example usage
    def example_search(query: str, max_results: int = 10, 
                      preferred_engines: List[str] = None) -> dict:
        """
        Search the web for information.
        
        Args:
            query: The search query string
            max_results: Maximum number of results to return
            preferred_engines: List of preferred search engines to use
        """
        pass
    
    # Create tool with strict validation
    try:
        tool = create_tool_from_function(example_search, strict_validation=True)
        print("✅ Tool created successfully:")
        print(json.dumps(tool, indent=2))
        
        # Validate for different providers
        is_valid, errors = validate_tools([tool], provider="gemini")
        if is_valid:
            print("\n✅ Valid for Gemini")
        else:
            print("\n❌ Validation errors for Gemini:")
            for error in errors:
                print(f"  - {error}")
                
    except ValueError as e:
        print(f"❌ Error creating tool: {e}")