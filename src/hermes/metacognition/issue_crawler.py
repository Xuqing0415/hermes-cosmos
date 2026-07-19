from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
import os
import re


@dataclass
class CommentData:
    id: int
    text: str
    author: str = "anonymous"
    sentiment: Optional[float] = None
    topics: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "author": self.author,
            "sentiment": self.sentiment,
            "topics": self.topics,
            "metadata": self.metadata
        }


class IssueCrawler:
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = data_dir
    
    def load_from_file(self, filename: str = "github_comments.txt") -> List[CommentData]:
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"[IssueCrawler] 文件不存在: {filepath}")
            return self._generate_fallback_comments()
        
        comments = []
        comment_id = 0
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                
                if not line or line.startswith('#') or line.startswith('##'):
                    continue
                
                match = re.match(r'^(\d+)\.\s*(.*)', line)
                if match:
                    comment_id = int(match.group(1))
                    text = match.group(2)
                else:
                    text = line
                
                if text:
                    comments.append(CommentData(
                        id=comment_id,
                        text=text,
                        author=f"user_{comment_id % 10}"
                    ))
        
        print(f"[IssueCrawler] 加载了 {len(comments)} 条评论")
        return comments
    
    def _generate_fallback_comments(self) -> List[CommentData]:
        fallback = [
            "The system didn't crash when limits were reached, just slowed down nicely.",
            "Error messages are very helpful and point directly to the problem.",
            "After using it a few times, it started generating exactly what I needed.",
            "Tests run too slow, takes too long to generate results.",
            "Documentation is lacking, hard to figure out how to use.",
            "API design is clean and consistent, easy to integrate.",
            "When tests fail, I have no idea why - error messages are cryptic.",
            "The system learned my patterns over time and got better.",
            "Completely freezes when memory is low, needs better handling.",
            "Recovered gracefully from network outage, impressive resilience.",
        ]
        
        return [CommentData(id=i+1, text=text) for i, text in enumerate(fallback)]
    
    def crawl_github(self, repo: str, token: str = "", max_comments: int = 100) -> List[CommentData]:
        try:
            from github import Github, RateLimitExceededException
        except ImportError:
            print("[IssueCrawler] PyGithub 未安装，使用样例数据")
            return self.load_from_file()
        
        comments = []
        
        try:
            g = Github(token) if token else Github()
            repo_obj = g.get_repo(repo)
            
            issues = repo_obj.get_issues(state="all")
            comment_id = 0
            
            for issue in issues:
                for comment in issue.get_comments():
                    if comment.body and len(comments) < max_comments:
                        comments.append(CommentData(
                            id=comment_id,
                            text=comment.body,
                            author=comment.user.login if comment.user else "anonymous"
                        ))
                        comment_id += 1
                
                if len(comments) >= max_comments:
                    break
            
            print(f"[IssueCrawler] 从 GitHub 爬取了 {len(comments)} 条评论")
        except RateLimitExceededException:
            print("[IssueCrawler] GitHub API 限流，使用样例数据")
            return self.load_from_file()
        except Exception as e:
            print(f"[IssueCrawler] 爬取失败: {e}，使用样例数据")
            return self.load_from_file()
        
        return comments