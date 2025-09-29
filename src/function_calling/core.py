"""
/**
 * @file core.py
 * @purpose Core class for the function calling and agentic loop framework with proper LLM response handling.
 * 
 * @dependencies
 * - openai: For API client (OpenRouter compatible).
 * - asyncio: For async operations.
 * - .models: Data models for messages and tools.
 * - .utils.xa_logger: For comprehensive LLM logging.
 *
 * @notes
 * - FIXED: Proper handling of OpenAI chat completion responses with function calling
 * - FIXED: Correct extraction of agent thoughts and tool calls from structured responses
 * - MODIFIED: Integrated `LLMLogger` via `enable_logging` flag in `__init__`.
 * - Implements multi-turn reasoning and tool execution.
 * - Includes token condensation for tool responses.
 */
"""

import os
import json
import time
import asyncio
from typing import List, Dict, Any, Callable, Tuple, Optional

import openai
from .models import Message

from dotenv import load_dotenv
load_dotenv()

from pathlib import Path

PROMPTS_PATH = Path.joinpath(Path(__file__).cwd(), 'src/prompts')

# Default system prompt to encourage explainability.
DEFAULT_SYSTEM_PROMPT = None

with open(PROMPTS_PATH/"executor.md") as f:
    DEFAULT_SYSTEM_PROMPT = f.read()

class LLMConfig:
    """Configuration for LLM calls"""
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is required")
        
        self.model = os.getenv("OPENROUTER_MODEL", "qwen/qwen3-next-80b-a3b-thinking")
        self.max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "32768"))
        self.temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.3"))
        self.timeout = int(os.getenv("OPENAI_TIMEOUT", "180"))
        self.max_retries = int(os.getenv("OPENAI_MAX_RETRIES", "3"))
        self.retry_delay = float(os.getenv("OPENAI_RETRY_DELAY", "1.0"))

# Global configuration instance
llm_config = LLMConfig()

def call_llm_with_tools(messages: List[Dict[str, Any]], tools: List[Dict[str, Any]] = None, 
                       temperature: Optional[float] = None, model: Optional[str] = None) -> Dict[str, Any]:
    """
    Production-ready LLM interface function with function calling support
    
    Args:
        messages: List of message objects for the conversation
        tools: Optional list of available tools/functions
        temperature: Optional temperature override (0.0-2.0)
        model: Optional model override
        
    Returns:
        Dict: The complete LLM response object with content and tool_calls
        
    Raises:
        Exception: If all retry attempts fail
    """
    client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=llm_config.api_key
    )
    
    # Use provided temperature or default
    temp = temperature if temperature is not None else llm_config.temperature
    
    # Prepare call parameters
    call_params = {
        "model": model if model else llm_config.model,
        "messages": messages,
        "max_tokens": llm_config.max_tokens,
        "temperature": temp,
        "timeout": llm_config.timeout
    }
    
    # Add tools if provided
    if tools:
        call_params["tools"] = tools
        call_params["tool_choice"] = "auto"
    
    # Retry logic with exponential backoff
    for attempt in range(llm_config.max_retries):
        try:
            response = client.chat.completions.create(**call_params)
            
            # Extract the response message
            response_message = response.choices[0].message
            
            # Convert to dictionary format for easier handling
            result = {
                "content": response_message.content,
                "tool_calls": []
            }
            
            # Extract tool calls if present
            if response_message.tool_calls:
                for tool_call in response_message.tool_calls:
                    result["tool_calls"].append({
                        "id": tool_call.id,
                        "type": tool_call.type,
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments
                        }
                    })
            
            return result
            
        except openai.RateLimitError as e:
            wait_time = llm_config.retry_delay * (2 ** attempt)  # Exponential backoff
            time.sleep(wait_time)
            if attempt == llm_config.max_retries - 1:
                raise Exception(f"Rate limit exceeded after {llm_config.max_retries} attempts") from e
                
        except openai.APITimeoutError as e:
            if attempt == llm_config.max_retries - 1:
                raise Exception(f"Timeout after {llm_config.max_retries} attempts") from e
            time.sleep(llm_config.retry_delay)
            
        except openai.APIConnectionError as e:
            if attempt == llm_config.max_retries - 1:
                raise Exception(f"Connection failed after {llm_config.max_retries} attempts") from e
            time.sleep(llm_config.retry_delay)
            
        except openai.AuthenticationError as e:
            raise Exception("OpenAI authentication failed") from e
            
        except Exception as e:
            if attempt == llm_config.max_retries - 1:
                raise Exception(f"LLM call failed after {llm_config.max_retries} attempts: {str(e)}") from e
            time.sleep(llm_config.retry_delay)
    
    raise Exception("All LLM call attempts failed")

def call_llm(prompt: str, system_prompt: Optional[str] = None, temperature: Optional[float] = None, model: Optional[str] = None) -> str:
    """
    Simple LLM interface function for backward compatibility
    
    Args:
        prompt: The user prompt/question
        system_prompt: Optional system message to guide behavior
        temperature: Optional temperature override (0.0-2.0)
        model: Optional model override
        
    Returns:
        str: The LLM response content
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    response = call_llm_with_tools(messages, tools=None, temperature=temperature, model=model)
    return response["content"] or ""

# Import LLMLogger
from .utils.xa_logger import enable_llm_logging

class FunctionCalling:
    """
    A class to handle function calling with an LLM via OpenRouter.
    This version implements an agentic loop with explainability and structured logging.
    """

    def __init__(
        self,
        model: str = "z-ai/glm-4.5",
        max_turns: int = 5,
        system_prompt: str = None,
        temperature: float = 0.0,
        max_tool_response_length: int = 4096,
        enable_logging: bool = False,
        log_dir: str = "llm_logs"
    ):
        """
        Initializes the FunctionCalling instance.

        Args:
            model: The model to use for chat completions.
            max_turns: The maximum number of tool-use cycles before stopping.
            system_prompt: The system prompt to guide the agent's behavior.
            temperature: The temperature parameter for the model's sampling (OpenRouter compatible).
            max_tool_response_length: The maximum character length for a tool's response before it's truncated in the message history. Set to None for no limit.
            enable_logging: If True, enables comprehensive LLM logging via LLMLogger.
            log_dir: The directory to store LLM logs.
        """
        self.model = model
        self.max_turns = max_turns
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.temperature = temperature
        self.max_tool_response_length = max_tool_response_length
        self.tools: List[Dict[str, Any]] = []
        self.functions: Dict[str, Callable] = {}
        
        # Logging setup - wrap the new call_llm_with_tools function
        self.logger = None
        if enable_logging:
            self._llm_call_func, self.logger = enable_llm_logging(call_llm_function=call_llm_with_tools, log_dir=log_dir)
        else:
            self._llm_call_func = call_llm_with_tools

    def register_tool(self, func: Callable):
        """Registers a function as a tool."""
        from .utils.utils import create_tool_from_function
        self.tools.append(create_tool_from_function(func))
        self.functions[func.__name__] = func

    def _condense_tool_response(self, content: str) -> str:
        """
        Truncates the tool response if it exceeds the configured maximum length.
        """
        if self.max_tool_response_length is None or len(content) <= self.max_tool_response_length:
            return content

        truncated_content = content[:self.max_tool_response_length]
        return (
            f"{truncated_content}\n"
            f"[... (Content truncated. Original length: {len(content)} characters)]"
        )

    def _extract_json_from_string(self, text: str) -> str:
        """
        Extracts a JSON object from a string that might contain extra text.
        Finds the first '{' and the last '}' to isolate the JSON part.
        """
        try:
            start_index = text.find('{')
            end_index = text.rfind('}')

            if start_index != -1 and end_index != -1 and end_index > start_index:
                return text[start_index:end_index+1]
        except Exception:
            pass
        return text

    async def run_async(
        self, user_prompt: str, tool_choice: str = "auto", external_tools: dict = None
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Runs the function calling process with a multi-step agentic loop.

        Returns:
            A tuple containing:
            - The final string response from the agent.
            - A structured execution log detailing each turn.
        """
        execution_log = []
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        for turn in range(self.max_turns):
            print(f"\n--- Agent Turn {turn + 1} ---")

            # Call LLM with tools
            try:
                response = self._llm_call_func(
                    messages=messages, 
                    tools=self.tools if self.tools else None,
                    temperature=self.temperature,
                    model=self.model
                )
                
                # Extract agent thought and tool calls from structured response
                agent_thought = response.get("content", "")
                tool_calls = response.get("tool_calls", [])
                
                # Add assistant message to conversation history
                assistant_message = {"role": "assistant", "content": agent_thought}
                if tool_calls:
                    # Convert tool calls back to OpenAI format for message history
                    assistant_message["tool_calls"] = []
                    for tc in tool_calls:
                        assistant_message["tool_calls"].append({
                            "id": tc["id"],
                            "type": tc["type"],
                            "function": {
                                "name": tc["function"]["name"],
                                "arguments": tc["function"]["arguments"]
                            }
                        })
                
                messages.append(assistant_message)

            except Exception as e:
                print(f"❌ Error during LLM call: {e}")
                return f"Error: {e}", execution_log

            # --- EXPLAINABILITY ---
            # Display the agent's thought process.
            if agent_thought:
                print(f"🤖 Agent's Thought: {agent_thought}")

            # If the model does not want to call a tool, the conversation is over.
            if not tool_calls:
                print("✅ Agent finished.")
                return agent_thought, execution_log

            # --- TOOL EXECUTION ---
            print(f"🛠️ Agent wants to use {len(tool_calls)} tool(s)...")
            tool_results = []
            
            for tool_call in tool_calls:
                function_name = tool_call["function"]["name"]
                
                # Handle JSON arguments robustly
                raw_arguments = tool_call["function"]["arguments"]
                cleaned_arguments = self._extract_json_from_string(raw_arguments)

                try:
                    function_args = json.loads(cleaned_arguments)
                except json.JSONDecodeError:
                    print(f"   - ⚠️ Warning: Failed to decode JSON arguments: '{cleaned_arguments}'")
                    function_args = {}

                print(f"   - Calling: {function_name}({json.dumps(function_args)})")

                # Find the function to call
                function_to_call = self.functions.get(function_name)

                if not function_to_call and external_tools:
                    for tool_instance in external_tools.values():
                        if hasattr(tool_instance, function_name):
                            function_to_call = getattr(tool_instance, function_name)
                            break

                if not function_to_call:
                    tool_result_content = f"Error: Tool '{function_name}' not found."
                else:
                    try:
                        if asyncio.iscoroutinefunction(function_to_call):
                            tool_result_content = await function_to_call(**function_args)
                        else:
                            tool_result_content = function_to_call(**function_args)
                    except Exception as e:
                        tool_result_content = f"Error executing tool '{function_name}': {e}"

                # Store full result for logging
                tool_results.append(str(tool_result_content))

                # Add condensed result to message history
                condensed_content = self._condense_tool_response(str(tool_result_content))

                messages.append({
                    "tool_call_id": tool_call["id"],
                    "role": "tool",
                    "name": function_name,
                    "content": condensed_content,
                })

            # --- LOGGING ---
            # Log the complete turn for observability.
            execution_log.append({
                'turn': turn + 1,
                'thought': agent_thought,
                'tool_calls': tool_calls,  # Already in dict format
                'tool_results': tool_results  # Full, uncondensed results
            })

        final_answer = "Agent stopped after reaching the maximum number of turns."
        return final_answer, execution_log