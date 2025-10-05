import subprocess
import json
from pathlib import Path
from datetime import datetime

def get_commit_info(repo_path='.'):
    """
    Retrieves comprehensive commit information including metadata and diff.
    
    Args:
        repo_path (str): Path to the Git repository. Defaults to current directory.
    
    Returns:
        dict: Structured commit data or error information.
    """
    try:
        # Get commit metadata
        metadata_result = subprocess.run(
            ['git', 'log', '-1', '--pretty=format:%H|%an|%ae|%ad|%s|%b'],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True
        )
        
        # Parse metadata
        parts = metadata_result.stdout.split('|', 5)
        commit_hash, author, email, date, subject, body = parts if len(parts) == 6 else (*parts, '')
        
        # Get diff with stats
        diff_result = subprocess.run(
            ['git', 'diff', 'HEAD~1', 'HEAD'],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True
        )
        
        # Get file statistics
        stats_result = subprocess.run(
            ['git', 'diff', '--stat', 'HEAD~1', 'HEAD'],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True
        )
        
        # Get list of changed files
        files_result = subprocess.run(
            ['git', 'diff', '--name-status', 'HEAD~1', 'HEAD'],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True
        )
        
        return {
            'success': True,
            'metadata': {
                'commit_hash': commit_hash,
                'author': author,
                'email': email,
                'date': date,
                'subject': subject,
                'body': body.strip()
            },
            'stats': stats_result.stdout,
            'files_changed': files_result.stdout,
            'diff': diff_result.stdout
        }
        
    except subprocess.CalledProcessError as e:
        return {'success': False, 'error': f"Git command failed: {e.stderr}"}
    except FileNotFoundError:
        return {'success': False, 'error': "Git not found. Ensure Git is installed."}
    except Exception as e:
        return {'success': False, 'error': f"Unexpected error: {str(e)}"}


def format_for_llm(commit_data):
    """
    Formats commit data into an LLM-friendly text format.
    
    Args:
        commit_data (dict): Structured commit data from get_commit_info.
    
    Returns:
        str: Formatted text ready for LLM processing.
    """
    if not commit_data.get('success'):
        return f"ERROR: {commit_data.get('error', 'Unknown error')}"
    
    meta = commit_data['metadata']
    
    output = f"""# Git Commit Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Commit Metadata
- **Commit Hash**: {meta['commit_hash']}
- **Author**: {meta['author']} <{meta['email']}>
- **Date**: {meta['date']}
- **Subject**: {meta['subject']}

## Commit Message
{meta['body'] if meta['body'] else '(No additional commit message)'}

## Files Changed Summary
```
{commit_data['stats']}
```

## Detailed File Changes
```
{commit_data['files_changed']}
```

## Full Diff
```diff
{commit_data['diff']}
```

---
## Instructions for LLM
Please analyze the above commit and create a progress.md report that includes:
1. A summary of what changed in this commit
2. The purpose/goal of these changes
3. Key modifications broken down by file or feature
4. Impact assessment (what functionality is affected)
5. Technical notes (any patterns, refactoring, or architectural changes)
6. Progress made towards project goals
"""
    return output


def save_commit_analysis(repo_path='.', output_file='commit_analysis.txt'):
    """
    Main function: Analyzes last commit and saves LLM-ready report to file.
    
    Args:
        repo_path (str): Path to the Git repository.
        output_file (str): Output filename for the analysis.
    
    Returns:
        str: Path to the saved file or error message.
    """
    commit_data = get_commit_info(repo_path)
    formatted_text = format_for_llm(commit_data)
    
    try:
        output_path = Path(output_file)
        output_path.write_text(formatted_text, encoding='utf-8')
        return f"Analysis saved to: {output_path.absolute()}"
    except Exception as e:
        return f"Failed to save file: {str(e)}"


# Usage examples
if __name__ == "__main__":
    # Example 1: Save analysis to default file
    result = save_commit_analysis()
    print(result)
    
    # Example 2: Custom repo and output file
    # result = save_commit_analysis(
    #     repo_path='/path/to/repo',
    #     output_file='llm_commit_analysis.txt'
    # )
    # print(result)
    
    # Example 3: Get data programmatically
    # commit_data = get_commit_info()
    # if commit_data['success']:
    #     print("Commit subject:", commit_data['metadata']['subject'])
    #     print("Files changed:", commit_data['files_changed'])