from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import urllib.request
import json
import time
import re


class TrendSource(Enum):
    GITHUB_TRENDING = "github_trending"
    PYPI_NEW = "pypi_new"
    ARXIV = "arxiv"


class TrendCategory(Enum):
    WEB_FRAMEWORK = "web_framework"
    DATABASE = "database"
    ML_AI = "ml_ai"
    DEVOPS = "devops"
    SECURITY = "security"
    PERFORMANCE = "performance"
    TESTING = "testing"
    OBSERVABILITY = "observability"
    OTHER = "other"


@dataclass
class TrendItem:
    source: TrendSource
    name: str
    description: str
    url: str
    stars: int = 0
    category: TrendCategory = TrendCategory.OTHER
    relevance_score: float = 0.0
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source.value,
            "name": self.name,
            "description": self.description,
            "url": self.url,
            "stars": self.stars,
            "category": self.category.value,
            "relevance_score": self.relevance_score,
            "tags": self.tags,
            "metadata": self.metadata
        }


@dataclass
class TrendAnalysisResult:
    trends: List[TrendItem] = field(default_factory=list)
    sources_scanned: int = 0
    total_items_found: int = 0
    top_recommendations: List[TrendItem] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "sources_scanned": self.sources_scanned,
            "total_items_found": self.total_items_found,
            "top_recommendations": [t.to_dict() for t in self.top_recommendations],
            "all_trends": [t.to_dict() for t in self.trends]
        }


class TrendAnalyzer:
    def __init__(self):
        self._timeout = 3
        self._user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    
    def analyze(self, audit_issues: List[Any] = None) -> TrendAnalysisResult:
        print("[趋势分析] 开始分析技术趋势...")
        
        trends = []
        sources_scanned = 0
        
        try:
            pypi_trends = self._fetch_pypi_trends()
            trends.extend(pypi_trends)
            sources_scanned += 1
            print(f"[趋势分析] 获取 PyPI 趋势: {len(pypi_trends)} 个包")
        except Exception as e:
            print(f"[趋势分析] 获取 PyPI 趋势失败: {e}")
        
        try:
            github_trends = self._fetch_github_trending()
            trends.extend(github_trends)
            sources_scanned += 1
            print(f"[趋势分析] 获取 GitHub 趋势: {len(github_trends)} 个项目")
        except Exception as e:
            print(f"[趋势分析] 获取 GitHub 趋势失败: {e}")
        
        if len(trends) == 0:
            trends = self._get_fallback_trends()
            print(f"[趋势分析] 使用备用趋势数据: {len(trends)} 个项目")
        
        if audit_issues:
            trends = self._score_relevance(trends, audit_issues)
        
        trends.sort(key=lambda t: -t.relevance_score)
        
        result = TrendAnalysisResult(
            trends=trends,
            sources_scanned=sources_scanned,
            total_items_found=len(trends),
            top_recommendations=trends[:10]
        )
        
        self._print_summary(result)
        
        return result
    
    def _get_fallback_trends(self) -> List[TrendItem]:
        return [
            TrendItem(
                source=TrendSource.PYPI_NEW,
                name="FastAPI",
                description="FastAPI framework, high performance, easy to learn, fast to code, ready for production",
                url="https://pypi.org/project/fastapi/",
                stars=65000,
                category=TrendCategory.WEB_FRAMEWORK,
                tags=["python", "async", "web"]
            ),
            TrendItem(
                source=TrendSource.PYPI_NEW,
                name="pydantic",
                description="Data validation using Python type hints",
                url="https://pypi.org/project/pydantic/",
                stars=25000,
                category=TrendCategory.WEB_FRAMEWORK,
                tags=["python", "validation"]
            ),
            TrendItem(
                source=TrendSource.GITHUB_TRENDING,
                name="ruff",
                description="An extremely fast Python linter and code formatter, written in Rust",
                url="https://github.com/astral-sh/ruff",
                stars=18000,
                category=TrendCategory.PERFORMANCE,
                tags=["rust", "python", "performance"]
            ),
            TrendItem(
                source=TrendSource.GITHUB_TRENDING,
                name="opentelemetry-python",
                description="OpenTelemetry Python SDK",
                url="https://github.com/open-telemetry/opentelemetry-python",
                stars=4500,
                category=TrendCategory.OBSERVABILITY,
                tags=["python", "observability", "monitoring"]
            ),
            TrendItem(
                source=TrendSource.PYPI_NEW,
                name="httpx",
                description="A next generation HTTP client for Python",
                url="https://pypi.org/project/httpx/",
                stars=12000,
                category=TrendCategory.WEB_FRAMEWORK,
                tags=["python", "async", "http"]
            ),
            TrendItem(
                source=TrendSource.PYPI_NEW,
                name="redis",
                description="Redis Python client",
                url="https://pypi.org/project/redis/",
                stars=10000,
                category=TrendCategory.DATABASE,
                tags=["python", "cache"]
            ),
            TrendItem(
                source=TrendSource.GITHUB_TRENDING,
                name="mypy",
                description="Optional static typing for Python",
                url="https://github.com/python/mypy",
                stars=15000,
                category=TrendCategory.TESTING,
                tags=["python", "type-checking"]
            ),
            TrendItem(
                source=TrendSource.PYPI_NEW,
                name="pytest",
                description="pytest: simple powerful testing with Python",
                url="https://pypi.org/project/pytest/",
                stars=10000,
                category=TrendCategory.TESTING,
                tags=["python", "testing"]
            ),
            TrendItem(
                source=TrendSource.GITHUB_TRENDING,
                name="py-cord",
                description="A Python wrapper for the Discord API with a focus on being easy to use",
                url="https://github.com/Pycord-Development/pycord",
                stars=7000,
                category=TrendCategory.OTHER,
                tags=["python", "api"]
            ),
            TrendItem(
                source=TrendSource.PYPI_NEW,
                name="kubernetes",
                description="Kubernetes Python client",
                url="https://pypi.org/project/kubernetes/",
                stars=5000,
                category=TrendCategory.DEVOPS,
                tags=["python", "kubernetes"]
            ),
        ]
    
    def _fetch_pypi_trends(self) -> List[TrendItem]:
        trends = []
        
        urls = [
            "https://pypi.org/rss/packages/new.xml",
            "https://pypistats.org/api/packages/pypy/recent"
        ]
        
        for url in urls:
            try:
                req = urllib.request.Request(url, headers={"User-Agent": self._user_agent})
                with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                    content = resp.read().decode()
                    
                    if url.endswith('.xml'):
                        packages = self._parse_pypi_rss(content)
                        for name, desc in packages[:20]:
                            category = self._classify_package(name, desc)
                            trends.append(TrendItem(
                                source=TrendSource.PYPI_NEW,
                                name=name,
                                description=desc,
                                url=f"https://pypi.org/project/{name}/",
                                category=category,
                                tags=self._extract_tags(name, desc)
                            ))
            except Exception:
                continue
        
        return trends
    
    def _parse_pypi_rss(self, xml_content: str) -> List[tuple]:
        packages = []
        import re
        
        items = re.findall(r'<item>(.*?)</item>', xml_content, re.DOTALL)
        for item in items[:20]:
            match = re.search(r'<title>([^<]+)</title>', item)
            desc_match = re.search(r'<description>([^<]+)</description>', item)
            
            if match:
                name = match.group(1).split(' ')[0]
                desc = desc_match.group(1) if desc_match else ""
                packages.append((name, desc))
        
        return packages
    
    def _fetch_github_trending(self) -> List[TrendItem]:
        trends = []
        
        url = "https://github.com/trending/python?since=daily"
        
        try:
            req = urllib.request.Request(url, headers={"User-Agent": self._user_agent})
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                content = resp.read().decode()
                
                # Split the page into per-article blocks first. Extracting all
                # fields from the whole page with one long regex triggers
                # catastrophic backtracking on the real GitHub HTML (multi-minute
                # CPU hang); per-block regexes stay bounded and fast.
                articles = re.findall(r'<article class="Box-row.*?</article>', content, re.DOTALL)
                
                for article in articles[:15]:
                    href_match = re.search(r'<h2[^>]*>\s*<a[^>]*href="([^"]+)"', article)
                    if not href_match:
                        continue
                    
                    name = self._parse_repo_name(article)
                    desc = self._parse_repo_description(article)
                    stars = self._parse_repo_stars(article)
                    category = self._classify_repo(name, desc)
                    
                    trends.append(TrendItem(
                        source=TrendSource.GITHUB_TRENDING,
                        name=name,
                        description=desc,
                        url=f"https://github.com{href_match.group(1)}",
                        stars=stars,
                        category=category,
                        tags=self._extract_tags(name, desc)
                    ))
        except Exception:
            pass
        
        return trends
    
    def _parse_repo_name(self, article: str) -> str:
        h2_match = re.search(r'<h2[^>]*>(.*?)</h2>', article, re.DOTALL)
        if not h2_match:
            return ""
        text = re.sub(r'<[^>]+>', ' ', h2_match.group(1))
        text = re.sub(r'\s+', ' ', text).strip()
        if "/" in text:
            return text.split("/")[-1].strip()
        return text
    
    def _parse_repo_description(self, article: str) -> str:
        match = re.search(r'<p[^>]*class="[^"]*col-9[^"]*"[^>]*>(.*?)</p>', article, re.DOTALL)
        if not match:
            return ""
        return re.sub(r'\s+', ' ', match.group(1)).strip()
    
    def _parse_repo_stars(self, article: str) -> int:
        match = re.search(r'href="([^"]*/stargazers)"[^>]*>(.*?)</a>', article, re.DOTALL)
        if not match:
            return 0
        inner = re.sub(r'<[^>]+>', ' ', match.group(2))
        return self._parse_stars(inner)
    
    def _parse_stars(self, stars_str: str) -> int:
        stars_str = stars_str.strip().replace(',', '')
        match = re.search(r'(\d+(?:\.\d+)?)\s*(k|m)?', stars_str, re.IGNORECASE)
        if match:
            value = float(match.group(1))
            suffix = match.group(2)
            if suffix == 'k':
                return int(value * 1000)
            elif suffix == 'm':
                return int(value * 1000000)
            return int(value)
        return 0
    
    def _classify_package(self, name: str, description: str) -> TrendCategory:
        text = f"{name} {description}".lower()
        
        if any(kw in text for kw in ['flask', 'fastapi', 'django', 'web', 'api', 'http']):
            return TrendCategory.WEB_FRAMEWORK
        if any(kw in text for kw in ['database', 'sql', 'orm', 'redis', 'mongodb']):
            return TrendCategory.DATABASE
        if any(kw in text for kw in ['ml', 'ai', 'machine learning', 'deep learning', 'neural']):
            return TrendCategory.ML_AI
        if any(kw in text for kw in ['kubernetes', 'docker', 'devops', 'ci/cd', 'deployment']):
            return TrendCategory.DEVOPS
        if any(kw in text for kw in ['security', 'auth', 'crypt', 'encryption']):
            return TrendCategory.SECURITY
        if any(kw in text for kw in ['performance', 'optimization', 'cache']):
            return TrendCategory.PERFORMANCE
        if any(kw in text for kw in ['test', 'pytest', 'unittest', 'mock']):
            return TrendCategory.TESTING
        
        return TrendCategory.OTHER
    
    def _classify_repo(self, name: str, description: str) -> TrendCategory:
        return self._classify_package(name, description)
    
    def _extract_tags(self, name: str, description: str) -> List[str]:
        text = f"{name} {description}".lower()
        tags = []
        
        if 'rust' in text:
            tags.append('rust')
        if 'python' in text:
            tags.append('python')
        if 'async' in text:
            tags.append('async')
        if 'grpc' in text:
            tags.append('grpc')
        if 'graphql' in text:
            tags.append('graphql')
        if 'observability' in text or 'monitoring' in text:
            tags.append('observability')
        if 'distributed' in text:
            tags.append('distributed')
        
        return tags
    
    def _score_relevance(self, trends: List[TrendItem], issues: List[Any]) -> List[TrendItem]:
        for trend in trends:
            score = 0.0
            
            for issue in issues:
                issue_text = f"{issue.title} {issue.description}".lower()
                
                if "security" in issue_text and trend.category == TrendCategory.SECURITY:
                    score += 0.5
                if "performance" in issue_text and trend.category == TrendCategory.PERFORMANCE:
                    score += 0.5
                if "web" in issue_text and trend.category == TrendCategory.WEB_FRAMEWORK:
                    score += 0.3
                if "testing" in issue_text and trend.category == TrendCategory.TESTING:
                    score += 0.3
                
                for tag in trend.tags:
                    if tag in issue_text:
                        score += 0.2
            
            if trend.stars > 10000:
                score += 0.3
            elif trend.stars > 1000:
                score += 0.1
            
            trend.relevance_score = min(score, 1.0)
        
        return trends
    
    def _print_summary(self, result: TrendAnalysisResult):
        print(f"[趋势分析] 扫描完成: {result.sources_scanned} 个源，{result.total_items_found} 个项目")
        print(f"\n[趋势分析] 推荐技术:")
        
        for i, trend in enumerate(result.top_recommendations[:5], 1):
            print(f"  {i}. {trend.name}")
            print(f"     类别: {trend.category.value}")
            print(f"     相关性: {trend.relevance_score:.2f}")
            print(f"     描述: {trend.description[:50]}...")
            print(f"     链接: {trend.url}")