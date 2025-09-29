"""
/**
 * @file url_context.py
 * @purpose Simple URL Context Tool for Planner Agent Integration.
 * 
 * @dependencies
 * - re, urllib.parse, socket, ssl, datetime, json, logging: Standard Python libraries.
 *
 * @notes
 * - MODIFIED: Implemented `__aenter__` and `__aexit__` to support the asynchronous context manager protocol,
 *   allowing it to be used in `async with` blocks alongside other asynchronous tools.
 * - Performs basic URL analysis, security scoring, and connectivity checks.
 */
"""

import re
import urllib.parse
import socket
import ssl
import datetime
from typing import Dict, List, Optional
import json
import logging


class URLContextTool:
    """Simple URL analyzer for security and structure checking."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    async def __aenter__(self):
        """
        Asynchronous context manager entry.
        No asynchronous setup is required for this tool, so it returns itself immediately.
        """
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Asynchronous context manager exit.
        No asynchronous cleanup is required for this tool.
        """
        pass

    def analyze_url(self, url: str) -> Dict:
        """Main method to analyze a URL and return context."""
        try:
            # Parse the URL
            parsed = self._parse_url(url)
            if 'error' in parsed:
                return parsed
                
            # Run all analyses
            result = {
                'url': url,
                'timestamp': datetime.datetime.now().isoformat(),
                'parsed': parsed,
                'security': self._check_security(parsed),
                'metadata': self._get_metadata(parsed),
                'validation': self._validate_url(url)
            }
            
            # Add connectivity check (optional)
            try:
                result['connectivity'] = self._check_connectivity(parsed)
            except:
                result['connectivity'] = {'status': 'check_failed'}
                
            return result
            
        except Exception as e:
            return {'error': f"Analysis failed: {str(e)}"}
    
    def _parse_url(self, url: str) -> Dict:
        """Break URL into parts."""
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
            
        try:
            parsed = urllib.parse.urlparse(url)
            domain_parts = parsed.hostname.split('.') if parsed.hostname else []
            
            return {
                'scheme': parsed.scheme,
                'hostname': parsed.hostname,
                'port': parsed.port,
                'path': parsed.path,
                'query': parsed.query,
                'domain': domain_parts[-2] if len(domain_parts) >= 2 else None,
                'tld': domain_parts[-1] if domain_parts else None,
                'is_secure': parsed.scheme == 'https'
            }
        except Exception as e:
            return {'error': f"Parse failed: {str(e)}"}
    
    def _check_security(self, parsed: Dict) -> Dict:
        """Check URL for security issues."""
        if 'error' in parsed:
            return {'error': 'Cannot check security of invalid URL'}
            
        security = {
            'score': 0,
            'issues': [],
            'is_secure': parsed['is_secure']
        }
        
        # Protocol check
        if parsed['is_secure']:
            security['score'] += 40
        else:
            security['issues'].append('Using HTTP instead of HTTPS')
            
        # Domain checks
        if parsed['hostname']:
            # Check for IP addresses
            if re.match(r'\d+\.\d+\.\d+\.\d+', parsed['hostname']):
                security['issues'].append('Using IP address instead of domain')
            else:
                security['score'] += 30
                
            # Domain length check
            if len(parsed['hostname']) > 50:
                security['issues'].append('Very long domain name')
            elif len(parsed['hostname']) >= 5:
                security['score'] += 30
                
        # Path checks
        if parsed['path']:
            suspicious_words = ['admin', 'login', 'password', 'secure']
            for word in suspicious_words:
                if word in parsed['path'].lower():
                    security['issues'].append(f'Path contains: {word}')
                    
        # Final score
        security['level'] = 'HIGH' if security['score'] >= 70 else 'MEDIUM' if security['score'] >= 40 else 'LOW'
        return security
    
    def _get_metadata(self, parsed: Dict) -> Dict:
        """Get URL metadata."""
        if 'error' in parsed:
            return {}
            
        metadata = {
            'has_query': bool(parsed['query']),
            'path_depth': len([p for p in parsed['path'].split('/') if p]),
            'is_root': parsed['path'] in ['', '/']
        }
        
        # Check for file extension
        if parsed['path'] and '.' in parsed['path']:
            ext = parsed['path'].split('.')[-1].lower()
            metadata['file_extension'] = ext
            metadata['file_type'] = self._get_file_type(ext)
            
        return metadata
    
    def _get_file_type(self, ext: str) -> str:
        """Get file type from extension."""
        types = {
            'html': 'Web Page', 'htm': 'Web Page', 'php': 'PHP Page',
            'pdf': 'PDF Document', 'doc': 'Word Document', 
            'jpg': 'Image', 'png': 'Image', 'gif': 'Image',
            'mp4': 'Video', 'mp3': 'Audio', 'zip': 'Archive',
            'json': 'Data File', 'xml': 'Data File', 'csv': 'Data File'
        }
        return types.get(ext, f'{ext.upper()} File')
    
    def _validate_url(self, url: str) -> Dict:
        """Check if URL format is valid."""
        validation = {'is_valid': False, 'issues': []}
        
        # Basic format check
        pattern = r'^https?://[^\s/$.?#].[^\s]*$'
        if re.match(pattern, url, re.IGNORECASE):
            validation['is_valid'] = True
        else:
            validation['issues'].append('Invalid URL format')
            
        return validation
    
    def _check_connectivity(self, parsed: Dict, timeout: int = 3) -> Dict:
        """Quick connectivity check."""
        if 'error' in parsed or not parsed['hostname']:
            return {'status': 'cannot_check'}
            
        try:
            host = parsed['hostname']
            port = parsed['port'] or (443 if parsed['is_secure'] else 80)
            
            sock = socket.create_connection((host, port), timeout)
            sock.close()
            
            return {'status': 'reachable'}
            
        except socket.timeout:
            return {'status': 'timeout'}
        except socket.gaierror:
            return {'status': 'dns_failed'}
        except ConnectionRefusedError:
            return {'status': 'connection_refused'}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def format_simple_report(self, analysis: Dict) -> str:
        """Create a simple text report."""
        if 'error' in analysis:
            return f"Error: {analysis['error']}"
            
        report = []
        report.append(f"URL: {analysis['url']}")
        
        # Security info
        if 'security' in analysis:
            sec = analysis['security']
            report.append(f"Security: {sec['level']} ({sec['score']}/100)")
            if sec['issues']:
                report.append(f"Issues: {', '.join(sec['issues'])}")
                
        # Basic info
        if 'parsed' in analysis:
            parsed = analysis['parsed']
            report.append(f"Protocol: {parsed['scheme']}")
            report.append(f"Domain: {parsed['hostname']}")
            
        # Connectivity
        if 'connectivity' in analysis:
            conn = analysis['connectivity']
            report.append(f"Status: {conn['status']}")
            
        return " | ".join(report)


# Integration function for the planner agent
def url_context_tool(url: str) -> Dict:
    """
    Main function to be called by the planner agent.
    Returns analysis of the given URL.
    """
    tool = URLContextTool()
    return tool.analyze_url(url)


# Example usage
if __name__ == "__main__":
    # Test the tool
    test_urls = [
        "https://www.google.com",
        "http://suspicious-site.com/login",
        "https://github.com/user/repo"
    ]
    
    tool = URLContextTool()
    for url in test_urls:
        print(f"\nTesting: {url}")
        result = tool.analyze_url(url)
        print(tool.format_simple_report(result))