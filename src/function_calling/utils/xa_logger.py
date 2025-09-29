"""
High Explainability LLM Logging System

This module provides comprehensive logging for LLM calls without modifying the main script.
It captures every interaction with detailed metadata for analysis and debugging.

/**
 * @file xa_logger.py  
 * @purpose Comprehensive logging system for LLM calls with support for both string and structured message formats
 * 
 * @dependencies
 * - json: For JSON serialization
 * - hashlib: For creating content hashes
 * - pathlib: For file system operations
 * - threading: For thread-safe logging
 *
 * @notes
 * - FIXED: Handle both string prompts and message dictionaries for function calling
 * - FIXED: Proper parameter extraction for different LLM call formats
 * - Maintains backward compatibility with original call_llm function
 * - Thread-safe logging with proper error handling
 */

Usage:
    from xa_logger import enable_llm_logging
    
    # Wrap your call_llm function
    logged_call_llm, logger = enable_llm_logging(original_call_llm)
    
    # Now all LLM calls will be automatically logged
"""

import json
import os
import time
import uuid
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional, Callable, Union, List
from pathlib import Path
import threading
from functools import wraps
import inspect

class LLMLogger:
    """
    Comprehensive logging system for LLM calls with high explainability
    """
    
    def __init__(self, log_dir: str = "llm_logs", enable_detailed_logging: bool = True):
        """
        Initialize the LLM Logger
        
        Args:
            log_dir: Directory to store log files
            enable_detailed_logging: Enable detailed metadata capture
        """
        self.log_dir = Path(log_dir)
        self.enable_detailed_logging = enable_detailed_logging
        self.session_id = str(uuid.uuid4())[:8]
        self.call_counter = 0
        self.lock = threading.Lock()
        
        # Create directory structure
        self.setup_directory_structure()
        
        # Initialize session metadata
        self.init_session()
    
    def setup_directory_structure(self):
        """Create organized directory structure for logs"""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # Main directories
        self.session_dir = self.log_dir / f"session_{timestamp}_{self.session_id}"
        self.raw_dir = self.session_dir / "raw_calls"
        self.structured_dir = self.session_dir / "structured_calls" 
        self.analysis_dir = self.session_dir / "analysis"
        self.summary_dir = self.session_dir / "summaries"
        
        # Create all directories
        for directory in [self.raw_dir, self.structured_dir, self.analysis_dir, self.summary_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def init_session(self):
        """Initialize session metadata file"""
        session_metadata = {
            "session_id": self.session_id,
            "start_time": datetime.now().isoformat(),
            "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}",
            "environment": dict(os.environ),
            "log_directory": str(self.session_dir),
            "detailed_logging_enabled": self.enable_detailed_logging
        }
        
        session_file = self.session_dir / "session_metadata.json"
        with open(session_file, 'w', encoding='utf-8') as f:
            json.dump(session_metadata, f, indent=2, ensure_ascii=False)
    
    def _extract_content_from_args(self, args: tuple, kwargs: dict) -> Dict[str, Any]:
        """
        Extract content from function arguments, handling both string and message formats
        
        Args:
            args: Positional arguments passed to LLM function
            kwargs: Keyword arguments passed to LLM function
            
        Returns:
            Dict containing extracted prompt, system_prompt, and other parameters
        """
        extracted = {
            "prompt": "",
            "system_prompt": "",
            "messages": None,
            "tools": None,
            "temperature": kwargs.get('temperature'),
            "model": kwargs.get('model'),
            "call_type": "unknown"
        }
        
        # Handle call_llm_with_tools format (messages as first arg or kwarg)
        if 'messages' in kwargs or (args and isinstance(args[0], list)):
            messages = kwargs.get('messages') or args[0]
            extracted["messages"] = messages
            extracted["call_type"] = "structured_messages"
            extracted["tools"] = kwargs.get('tools')
            
            # Extract prompt and system prompt from messages
            for msg in messages:
                if isinstance(msg, dict):
                    if msg.get('role') == 'system':
                        extracted["system_prompt"] = msg.get('content', '')
                    elif msg.get('role') == 'user':
                        extracted["prompt"] = msg.get('content', '')
        
        # Handle call_llm format (prompt as first arg or kwarg)  
        elif args and isinstance(args[0], str):
            extracted["prompt"] = args[0]
            extracted["system_prompt"] = kwargs.get('system_prompt', '') or (args[1] if len(args) > 1 else '')
            extracted["call_type"] = "simple_string"
        elif 'prompt' in kwargs:
            extracted["prompt"] = kwargs['prompt']
            extracted["system_prompt"] = kwargs.get('system_prompt', '')
            extracted["call_type"] = "simple_string"
        
        return extracted
    
    def _create_content_hash(self, content: Union[str, List, Dict, None]) -> str:
        """
        Create MD5 hash of content, handling different data types safely
        
        Args:
            content: Content to hash (string, list, dict, or None)
            
        Returns:
            MD5 hash string, or empty string if content is None
        """
        if content is None:
            return ""
        
        try:
            if isinstance(content, str):
                return hashlib.md5(content.encode('utf-8')).hexdigest()
            else:
                # Convert non-string content to JSON string first
                content_str = json.dumps(content, sort_keys=True, ensure_ascii=False)
                return hashlib.md5(content_str.encode('utf-8')).hexdigest()
        except Exception as e:
            # Fallback: create hash of string representation
            try:
                return hashlib.md5(str(content).encode('utf-8')).hexdigest()
            except:
                return f"hash_error_{uuid.uuid4().hex[:8]}"
    
    def wrap_llm_call(self, original_call_llm: Callable) -> Callable:
        """
        Wrap the original call_llm function with logging capability
        
        Args:
            original_call_llm: The original call_llm function to wrap
            
        Returns:
            Wrapped function that logs all calls
        """
        @wraps(original_call_llm)
        def logged_call_llm(*args, **kwargs):
            with self.lock:
                self.call_counter += 1
                call_id = f"call_{self.call_counter:04d}_{uuid.uuid4().hex[:8]}"
            
            # Pre-call logging
            start_time = time.time()
            call_metadata = self.capture_pre_call_metadata(call_id, args, kwargs)
            
            try:
                # Make the actual LLM call
                response = original_call_llm(*args, **kwargs)
                
                # Post-call logging (success)
                end_time = time.time()
                self.log_successful_call(call_id, call_metadata, response, start_time, end_time, args, kwargs)
                
                return response
                
            except Exception as e:
                # Post-call logging (error)
                end_time = time.time()
                self.log_failed_call(call_id, call_metadata, e, start_time, end_time, args, kwargs)
                raise  # Re-raise the exception
        
        return logged_call_llm
    
    def capture_pre_call_metadata(self, call_id: str, args: tuple, kwargs: dict) -> Dict[str, Any]:
        """Capture metadata before making the LLM call"""
        
        # Get caller information
        caller_frame = inspect.currentframe().f_back.f_back.f_back
        caller_info = {
            "filename": caller_frame.f_code.co_filename,
            "function_name": caller_frame.f_code.co_name,
            "line_number": caller_frame.f_lineno,
            "local_variables": list(caller_frame.f_locals.keys()) if self.enable_detailed_logging else []
        }
        
        # Extract content using the improved method
        content_info = self._extract_content_from_args(args, kwargs)
        
        metadata = {
            "call_id": call_id,
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "call_number": self.call_counter,
            "caller_info": caller_info,
            "call_type": content_info["call_type"],
            "prompt_hash": self._create_content_hash(content_info["prompt"]),
            "prompt_length": len(content_info["prompt"]) if content_info["prompt"] else 0,
            "system_prompt_hash": self._create_content_hash(content_info["system_prompt"]),
            "system_prompt_length": len(content_info["system_prompt"]) if content_info["system_prompt"] else 0,
            "messages_hash": self._create_content_hash(content_info["messages"]),
            "messages_count": len(content_info["messages"]) if content_info["messages"] else 0,
            "has_tools": content_info["tools"] is not None,
            "tools_count": len(content_info["tools"]) if content_info["tools"] else 0,
            "temperature": content_info["temperature"],
            "model": content_info["model"],
            "parameters": {
                "args_count": len(args),
                "kwargs_keys": list(kwargs.keys())
            }
        }
        
        return metadata
    
    def log_successful_call(self, call_id: str, metadata: Dict[str, Any], 
                          response: Union[str, Dict], start_time: float, end_time: float, 
                          args: tuple, kwargs: dict):
        """Log a successful LLM call with full details"""
        
        # Calculate metrics
        duration = end_time - start_time
        
        # Extract content for analysis
        content_info = self._extract_content_from_args(args, kwargs)
        
        # Handle different response formats
        if isinstance(response, dict):
            response_content = response.get("content", "")
            response_str = json.dumps(response, ensure_ascii=False)
            has_tool_calls = bool(response.get("tool_calls"))
            tool_calls_count = len(response.get("tool_calls", []))
        else:
            response_content = str(response)
            response_str = response_content
            has_tool_calls = False
            tool_calls_count = 0
        
        # Response analysis
        response_analysis = {
            "response_length": len(response_str),
            "response_hash": self._create_content_hash(response_str),
            "response_lines": len(response_str.split('\n')),
            "response_words": len(response_str.split()),
            "response_format": "structured" if isinstance(response, dict) else "string",
            "has_tool_calls": has_tool_calls,
            "tool_calls_count": tool_calls_count,
            "contains_code": '```' in response_str or 'def ' in response_str or 'class ' in response_str,
            "contains_numbers": any(char.isdigit() for char in response_str),
            "contains_list": any(marker in response_str for marker in ['1.', '2.', '-', '*'])
        }
        
        # Estimate tokens (rough approximation)
        input_text = content_info["prompt"] + content_info["system_prompt"]
        if content_info["messages"]:
            input_text = json.dumps(content_info["messages"])
        
        # Complete log entry
        log_entry = {
            **metadata,
            "status": "success",
            "duration_seconds": round(duration, 3),
            "response_analysis": response_analysis,
            "end_time": datetime.now().isoformat(),
            "estimated_tokens": {
                "input_tokens": len(input_text.split()),
                "output_tokens": len(response_str.split()),
                "total_tokens": len(input_text.split()) + len(response_str.split())
            }
        }
        
        # Save raw interaction
        self.save_raw_interaction(call_id, content_info, response_str, log_entry)
        
        # Save structured log
        self.save_structured_log(call_id, log_entry)
        
        # Update session summary
        self.update_session_summary(log_entry)
    
    def log_failed_call(self, call_id: str, metadata: Dict[str, Any], 
                       error: Exception, start_time: float, end_time: float,
                       args: tuple, kwargs: dict):
        """Log a failed LLM call with error details"""
        
        duration = end_time - start_time
        content_info = self._extract_content_from_args(args, kwargs)
        
        # Error analysis
        error_analysis = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "error_module": getattr(error, '__module__', None),
            "is_rate_limit": "rate limit" in str(error).lower(),
            "is_timeout": "timeout" in str(error).lower(),
            "is_auth_error": "auth" in str(error).lower(),
            "is_encoding_error": "encode" in str(error).lower()
        }
        
        log_entry = {
            **metadata,
            "status": "failed",
            "duration_seconds": round(duration, 3),
            "error_analysis": error_analysis,
            "end_time": datetime.now().isoformat()
        }
        
        # Save raw interaction (with error)
        self.save_raw_interaction(call_id, content_info, f"ERROR: {error}", log_entry)
        
        # Save structured log
        self.save_structured_log(call_id, log_entry)
        
        # Update session summary
        self.update_session_summary(log_entry)
    
    def save_raw_interaction(self, call_id: str, content_info: Dict[str, Any], 
                           response: str, metadata: Dict[str, Any]):
        """Save raw interaction in human-readable format"""
        
        # Format messages if available
        messages_display = ""
        if content_info["messages"]:
            messages_display = "\n=== MESSAGES ===\n"
            for i, msg in enumerate(content_info["messages"]):
                messages_display += f"Message {i+1} ({msg.get('role', 'unknown')}):\n{msg.get('content', '')}\n\n"
        
        # Format tools if available
        tools_display = ""
        if content_info["tools"]:
            tools_display = f"\n=== TOOLS ({len(content_info['tools'])}) ===\n"
            for tool in content_info["tools"]:
                tool_name = tool.get('function', {}).get('name', 'unknown')
                tools_display += f"- {tool_name}\n"
        
        raw_content = f"""
=== LLM CALL: {call_id} ===
Timestamp: {metadata['timestamp']}
Duration: {metadata.get('duration_seconds', 'N/A')}s
Status: {metadata['status']}
Call Type: {metadata.get('call_type', 'unknown')}
Caller: {metadata['caller_info']['function_name']} ({metadata['caller_info']['filename']}:{metadata['caller_info']['line_number']})

=== SYSTEM PROMPT ===
{content_info['system_prompt'] or '[No system prompt]'}

=== USER PROMPT ===
{content_info['prompt'] or '[No simple prompt]'}

{messages_display}

{tools_display}

=== RESPONSE ===
{response}

=== METADATA ===
{json.dumps(metadata, indent=2, ensure_ascii=False)}

{'='*80}
"""
        
        raw_file = self.raw_dir / f"{call_id}.txt"
        with open(raw_file, 'w', encoding='utf-8') as f:
            f.write(raw_content)
    
    def save_structured_log(self, call_id: str, log_entry: Dict[str, Any]):
        """Save structured log entry as JSON"""
        
        structured_file = self.structured_dir / f"{call_id}.json"
        with open(structured_file, 'w', encoding='utf-8') as f:
            json.dump(log_entry, f, indent=2, ensure_ascii=False)
    
    def update_session_summary(self, log_entry: Dict[str, Any]):
        """Update running session summary"""
        
        summary_file = self.summary_dir / "session_summary.json"
        
        # Load existing summary or create new
        if summary_file.exists():
            with open(summary_file, 'r', encoding='utf-8') as f:
                summary = json.load(f)
        else:
            summary = {
                "session_id": self.session_id,
                "total_calls": 0,
                "successful_calls": 0,
                "failed_calls": 0,
                "function_calling_calls": 0,
                "simple_string_calls": 0,
                "total_duration": 0,
                "total_estimated_tokens": 0,
                "error_types": {},
                "caller_functions": {},
                "last_updated": None
            }
        
        # Update summary
        summary["total_calls"] += 1
        summary["total_duration"] += log_entry.get("duration_seconds", 0)
        summary["last_updated"] = datetime.now().isoformat()
        
        # Track call types
        call_type = log_entry.get("call_type", "unknown")
        if call_type == "structured_messages":
            summary["function_calling_calls"] += 1
        elif call_type == "simple_string":
            summary["simple_string_calls"] += 1
        
        if log_entry["status"] == "success":
            summary["successful_calls"] += 1
            summary["total_estimated_tokens"] += log_entry.get("estimated_tokens", {}).get("total_tokens", 0)
        else:
            summary["failed_calls"] += 1
            error_type = log_entry.get("error_analysis", {}).get("error_type", "Unknown")
            summary["error_types"][error_type] = summary["error_types"].get(error_type, 0) + 1
        
        # Track caller functions
        caller_func = log_entry["caller_info"]["function_name"]
        summary["caller_functions"][caller_func] = summary["caller_functions"].get(caller_func, 0) + 1
        
        # Save updated summary
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    
    def generate_analysis_report(self):
        """Generate comprehensive analysis report"""
        
        analysis_file = self.analysis_dir / f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        # Load all structured logs
        all_calls = []
        for log_file in self.structured_dir.glob("*.json"):
            with open(log_file, 'r', encoding='utf-8') as f:
                all_calls.append(json.load(f))
        
        if not all_calls:
            return
        
        # Generate analysis
        report_content = self._generate_markdown_report(all_calls)
        
        with open(analysis_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"📊 Analysis report generated: {analysis_file}")
    
    def _generate_markdown_report(self, all_calls: list) -> str:
        """Generate markdown analysis report"""
        
        successful_calls = [call for call in all_calls if call["status"] == "success"]
        failed_calls = [call for call in all_calls if call["status"] == "failed"]
        
        # Calculate statistics
        total_duration = sum(call.get("duration_seconds", 0) for call in all_calls)
        avg_duration = total_duration / len(all_calls) if all_calls else 0
        
        total_tokens = sum(call.get("estimated_tokens", {}).get("total_tokens", 0) for call in successful_calls)
        
        # Call type analysis
        call_types = {}
        for call in all_calls:
            call_type = call.get("call_type", "unknown")
            call_types[call_type] = call_types.get(call_type, 0) + 1
        
        # Generate report
        report = f"""# LLM Call Analysis Report

## Session Overview
- **Session ID**: {self.session_id}
- **Total Calls**: {len(all_calls)}
- **Successful Calls**: {len(successful_calls)} ({len(successful_calls)/len(all_calls)*100:.1f}%)
- **Failed Calls**: {len(failed_calls)} ({len(failed_calls)/len(all_calls)*100:.1f}%)
- **Total Duration**: {total_duration:.2f} seconds
- **Average Call Duration**: {avg_duration:.2f} seconds
- **Estimated Total Tokens**: {total_tokens:,}

## Call Type Distribution
"""
        
        for call_type, count in sorted(call_types.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(all_calls)) * 100
            report += f"- **{call_type}**: {count} calls ({percentage:.1f}%)\n"
        
        # Add duration analysis
        durations = [call.get("duration_seconds", 0) for call in successful_calls]
        if durations:
            report += f"""
## Performance Analysis

### Duration Distribution
- **Min Duration**: {min(durations):.2f}s
- **Max Duration**: {max(durations):.2f}s
- **Median Duration**: {sorted(durations)[len(durations)//2]:.2f}s
"""
        
        # Error analysis
        if failed_calls:
            error_types = {}
            for call in failed_calls:
                error_type = call.get("error_analysis", {}).get("error_type", "Unknown")
                error_types[error_type] = error_types.get(error_type, 0) + 1
            
            report += f"""
## Error Analysis

### Error Types
"""
            for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
                report += f"- **{error_type}**: {count} occurrences\n"
        
        # Function calling analysis
        function_calls = [call for call in successful_calls if call.get("response_analysis", {}).get("has_tool_calls")]
        if function_calls:
            total_tool_calls = sum(call.get("response_analysis", {}).get("tool_calls_count", 0) for call in function_calls)
            report += f"""
## Function Calling Analysis
- **Calls with Tool Usage**: {len(function_calls)}
- **Total Tool Calls Made**: {total_tool_calls}
- **Average Tools per Call**: {total_tool_calls/len(function_calls):.1f}
"""
        
        # Caller analysis
        caller_stats = {}
        for call in all_calls:
            caller = call["caller_info"]["function_name"]
            caller_stats[caller] = caller_stats.get(caller, 0) + 1
        
        report += f"""
## Caller Analysis

### Function Call Distribution
"""
        for caller, count in sorted(caller_stats.items(), key=lambda x: x[1], reverse=True):
            report += f"- **{caller}()**: {count} calls\n"
        
        report += f"""
## Files Generated
- **Raw Interactions**: {len(list(self.raw_dir.glob('*.txt')))} files
- **Structured Logs**: {len(list(self.structured_dir.glob('*.json')))} files
- **Log Directory**: `{self.session_dir}`

---
*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        return report

# Convenience function for easy integration
def enable_llm_logging(call_llm_function: Callable, log_dir: str = "llm_logs") -> tuple:
    """
    Enable comprehensive logging for an LLM call function
    
    Args:
        call_llm_function: The original call_llm function
        log_dir: Directory to store logs
        
    Returns:
        tuple: (wrapped_function, logger_instance)
    """
    logger = LLMLogger(log_dir=log_dir)
    wrapped_function = logger.wrap_llm_call(call_llm_function)
    return wrapped_function, logger

# Example usage
if __name__ == "__main__":
    # Example of how to integrate with existing code
    def dummy_call_llm_with_tools(messages, tools=None, temperature=None, model=None):
        """Dummy LLM function for testing with function calling"""
        time.sleep(0.1)  # Simulate API call
        return {
            "content": f"Response to messages: {len(messages)} messages",
            "tool_calls": [{"id": "test", "function": {"name": "test_tool", "arguments": "{}"}}] if tools else []
        }
    
    # Enable logging
    logged_call_llm, logger = enable_llm_logging(dummy_call_llm_with_tools)
    
    # Test function calling format
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "What's the weather?"}
    ]
    tools = [{"function": {"name": "get_weather"}}]
    
    response1 = logged_call_llm(messages=messages, tools=tools)
    response2 = logged_call_llm(messages=messages)
    
    # Generate analysis report
    logger.generate_analysis_report()
    
    print(f"Logging enabled! Check logs in: {logger.session_dir}")