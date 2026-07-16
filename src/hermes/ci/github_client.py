"""
GitHub Client - Interact with GitHub API for PR comments and data
"""

import os
from typing import Dict, Optional, Any, List
import structlog

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    requests = None
    HAS_REQUESTS = False

logger = structlog.get_logger()


class GitHubClient:
    """
    Client for interacting with GitHub API.
    
    Supports:
    - Getting PR information
    - Posting comments on PRs
    - Getting commit diffs
    - Creating issues
    """
    
    def __init__(self, token: Optional[str] = None, repo_owner: str = "", repo_name: str = ""):
        self.token = token or os.environ.get("GITHUB_TOKEN")
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.base_url = "https://api.github.com"
        
        if self.token:
            self.headers = {
                "Authorization": f"token {self.token}",
                "Accept": "application/vnd.github.v3+json"
            }
        else:
            self.headers = {
                "Accept": "application/vnd.github.v3+json"
            }
        
        logger.info("GitHub client initialized", has_token=bool(self.token))
    
    def get_pr_info(self, pr_number: int) -> Optional[Dict[str, Any]]:
        """
        Get information about a PR.
        
        Args:
            pr_number: PR number
        
        Returns:
            Dictionary with PR information
        """
        if not HAS_REQUESTS:
            logger.warning("requests library not available")
            return None
        
        url = f"{self.base_url}/repos/{self.repo_owner}/{self.repo_name}/pulls/{pr_number}"
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error("Failed to get PR info", pr_number=pr_number, error=str(e))
            return None
    
    def post_pr_comment(self, pr_number: int, body: str) -> Optional[Dict[str, Any]]:
        """
        Post a comment on a PR.
        
        Args:
            pr_number: PR number
            body: Comment body (Markdown)
        
        Returns:
            Dictionary with comment information
        """
        if not HAS_REQUESTS:
            logger.warning("requests library not available")
            return None
        
        url = f"{self.base_url}/repos/{self.repo_owner}/{self.repo_name}/issues/{pr_number}/comments"
        
        try:
            response = requests.post(
                url,
                headers=self.headers,
                json={"body": body}
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error("Failed to post PR comment", pr_number=pr_number, error=str(e))
            return None
    
    def update_pr_comment(self, comment_id: int, body: str) -> Optional[Dict[str, Any]]:
        """
        Update an existing PR comment.
        
        Args:
            comment_id: ID of the comment to update
            body: New comment body (Markdown)
        
        Returns:
            Dictionary with updated comment information
        """
        if not HAS_REQUESTS:
            logger.warning("requests library not available")
            return None
        
        url = f"{self.base_url}/repos/{self.repo_owner}/{self.repo_name}/issues/comments/{comment_id}"
        
        try:
            response = requests.patch(
                url,
                headers=self.headers,
                json={"body": body}
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error("Failed to update PR comment", comment_id=comment_id, error=str(e))
            return None
    
    def get_pr_comments(self, pr_number: int) -> List[Dict[str, Any]]:
        """
        Get all comments on a PR.
        
        Args:
            pr_number: PR number
        
        Returns:
            List of comment dictionaries
        """
        if not HAS_REQUESTS:
            logger.warning("requests library not available")
            return []
        
        url = f"{self.base_url}/repos/{self.repo_owner}/{self.repo_name}/issues/{pr_number}/comments"
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error("Failed to get PR comments", pr_number=pr_number, error=str(e))
            return []
    
    def get_commit_diff(self, commit_sha: str) -> Optional[str]:
        """
        Get the diff for a commit.
        
        Args:
            commit_sha: Commit SHA
        
        Returns:
            Diff as string
        """
        if not HAS_REQUESTS:
            logger.warning("requests library not available")
            return None
        
        url = f"{self.base_url}/repos/{self.repo_owner}/{self.repo_name}/commits/{commit_sha}"
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            return data.get("files", [])
        except requests.exceptions.RequestException as e:
            logger.error("Failed to get commit diff", commit_sha=commit_sha, error=str(e))
            return None
    
    def create_issue(
        self,
        title: str,
        body: str,
        labels: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Create a GitHub issue.
        
        Args:
            title: Issue title
            body: Issue body (Markdown)
            labels: List of labels to apply
        
        Returns:
            Dictionary with issue information
        """
        if not HAS_REQUESTS:
            logger.warning("requests library not available")
            return None
        
        url = f"{self.base_url}/repos/{self.repo_owner}/{self.repo_name}/issues"
        
        try:
            response = requests.post(
                url,
                headers=self.headers,
                json={
                    "title": title,
                    "body": body,
                    "labels": labels or []
                }
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error("Failed to create issue", title=title, error=str(e))
            return None
    
    def get_repo_info(self) -> Optional[Dict[str, Any]]:
        """
        Get repository information.
        
        Returns:
            Dictionary with repository information
        """
        if not HAS_REQUESTS:
            logger.warning("requests library not available")
            return None
        
        url = f"{self.base_url}/repos/{self.repo_owner}/{self.repo_name}"
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error("Failed to get repo info", error=str(e))
            return None
    
    def set_repo(self, owner: str, name: str):
        """
        Set the repository to work with.
        
        Args:
            owner: Repository owner
            name: Repository name
        """
        self.repo_owner = owner
        self.repo_name = name
        logger.info("Repository set", owner=owner, name=name)
    
    def is_available(self) -> bool:
        """
        Check if the GitHub API is available.
        
        Returns:
            True if available, False otherwise
        """
        if not HAS_REQUESTS:
            return False
        
        if not self.token:
            return False
        
        try:
            response = requests.get(f"{self.base_url}/user", headers=self.headers)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
