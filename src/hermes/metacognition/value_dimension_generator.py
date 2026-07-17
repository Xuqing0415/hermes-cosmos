from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

from .topic_extractor import TopicResult
from .sentiment_analyzer import SentimentResult


class ValueDimensionType(Enum):
    EXPERIENCE = "experience"
    RELIABILITY = "reliability"
    LEARNING = "learning"
    PERFORMANCE = "performance"
    USABILITY = "usability"
    DEBUGGABILITY = "debuggability"
    ADAPTABILITY = "adaptability"
    RESILIENCE = "resilience"
    DOCUMENTATION = "documentation"
    API_DESIGN = "api_design"


@dataclass
class ValueDimension:
    id: int
    name: str
    type: ValueDimensionType
    description: str
    keywords: List[str]
    positive_examples: List[str] = field(default_factory=list)
    negative_examples: List[str] = field(default_factory=list)
    importance_score: float = 0.0
    sentiment_balance: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type.value,
            "description": self.description,
            "keywords": self.keywords,
            "positive_examples": self.positive_examples,
            "negative_examples": self.negative_examples,
            "importance_score": round(self.importance_score, 4),
            "sentiment_balance": round(self.sentiment_balance, 4)
        }


class ValueDimensionGenerator:
    DIMENSION_TEMPLATES = {
        ValueDimensionType.EXPERIENCE: {
            "name": "User Experience",
            "description": "How users perceive and interact with the system, including intuitiveness and overall satisfaction.",
            "keywords": ["intuitive", "easy", "helpful", "confusing", "ux", "experience"]
        },
        ValueDimensionType.RELIABILITY: {
            "name": "Reliability",
            "description": "The system's ability to perform consistently and correctly over time without failures.",
            "keywords": ["stable", "reliable", "crash", "fail", "flaky", "broken"]
        },
        ValueDimensionType.LEARNING: {
            "name": "Convergent Learning",
            "description": "The system's ability to adapt and improve based on user interactions and feedback over time.",
            "keywords": ["learn", "learned", "adapt", "adaptive", "improved", "got better"]
        },
        ValueDimensionType.PERFORMANCE: {
            "name": "Performance",
            "description": "System responsiveness, speed, and resource efficiency during operation.",
            "keywords": ["slow", "fast", "quick", "performance", "speed", "response"]
        },
        ValueDimensionType.USABILITY: {
            "name": "Usability",
            "description": "Ease of use and accessibility of the system's interfaces and APIs.",
            "keywords": ["usability", "intuitive", "confusing", "easy", "simple", "complex"]
        },
        ValueDimensionType.DEBUGGABILITY: {
            "name": "Explainable Failures",
            "description": "The clarity and helpfulness of error messages and debugging information when things go wrong.",
            "keywords": ["error", "debug", "cryptic", "helpful", "message", "trace", "stack"]
        },
        ValueDimensionType.ADAPTABILITY: {
            "name": "Adaptability",
            "description": "The system's ability to adjust its behavior based on user feedback and changing conditions.",
            "keywords": ["adapt", "adaptive", "feedback", "learn", "adjust", "evolve"]
        },
        ValueDimensionType.RESILIENCE: {
            "name": "Gently Degradable",
            "description": "The system's ability to degrade gracefully under stress rather than failing abruptly.",
            "keywords": ["graceful", "degrade", "freeze", "crash", "recover", "resilience"]
        },
        ValueDimensionType.DOCUMENTATION: {
            "name": "Documentation Quality",
            "description": "The completeness, clarity, and helpfulness of system documentation and guides.",
            "keywords": ["documentation", "docs", "guide", "sparse", "lacking", "examples"]
        },
        ValueDimensionType.API_DESIGN: {
            "name": "API Design",
            "description": "The quality of the system's application programming interface, including consistency and ease of integration.",
            "keywords": ["api", "design", "consistent", "clean", "breaking", "version"]
        }
    }
    
    KEYWORD_TO_DIMENSION = {
        "slow": ValueDimensionType.PERFORMANCE,
        "fast": ValueDimensionType.PERFORMANCE,
        "performance": ValueDimensionType.PERFORMANCE,
        "speed": ValueDimensionType.PERFORMANCE,
        "quick": ValueDimensionType.PERFORMANCE,
        "response": ValueDimensionType.PERFORMANCE,
        
        "crash": ValueDimensionType.RELIABILITY,
        "freeze": ValueDimensionType.RESILIENCE,
        "fail": ValueDimensionType.RELIABILITY,
        "failed": ValueDimensionType.RELIABILITY,
        "flaky": ValueDimensionType.RELIABILITY,
        "stable": ValueDimensionType.RELIABILITY,
        "reliable": ValueDimensionType.RELIABILITY,
        "broken": ValueDimensionType.RELIABILITY,
        
        "graceful": ValueDimensionType.RESILIENCE,
        "degrade": ValueDimensionType.RESILIENCE,
        "degraded": ValueDimensionType.RESILIENCE,
        "recover": ValueDimensionType.RESILIENCE,
        "resilience": ValueDimensionType.RESILIENCE,
        "resilient": ValueDimensionType.RESILIENCE,
        
        "error": ValueDimensionType.DEBUGGABILITY,
        "debug": ValueDimensionType.DEBUGGABILITY,
        "cryptic": ValueDimensionType.DEBUGGABILITY,
        "helpful": ValueDimensionType.DEBUGGABILITY,
        "message": ValueDimensionType.DEBUGGABILITY,
        "trace": ValueDimensionType.DEBUGGABILITY,
        "stack": ValueDimensionType.DEBUGGABILITY,
        
        "learn": ValueDimensionType.LEARNING,
        "learned": ValueDimensionType.LEARNING,
        "adapt": ValueDimensionType.ADAPTABILITY,
        "adaptive": ValueDimensionType.ADAPTABILITY,
        "improved": ValueDimensionType.LEARNING,
        "got better": ValueDimensionType.LEARNING,
        
        "intuitive": ValueDimensionType.USABILITY,
        "easy": ValueDimensionType.USABILITY,
        "confusing": ValueDimensionType.USABILITY,
        "simple": ValueDimensionType.USABILITY,
        "complex": ValueDimensionType.USABILITY,
        
        "documentation": ValueDimensionType.DOCUMENTATION,
        "docs": ValueDimensionType.DOCUMENTATION,
        "guide": ValueDimensionType.DOCUMENTATION,
        "sparse": ValueDimensionType.DOCUMENTATION,
        "lacking": ValueDimensionType.DOCUMENTATION,
        "examples": ValueDimensionType.DOCUMENTATION,
        
        "api": ValueDimensionType.API_DESIGN,
        "design": ValueDimensionType.API_DESIGN,
        "consistent": ValueDimensionType.API_DESIGN,
        "clean": ValueDimensionType.API_DESIGN,
        "breaking": ValueDimensionType.API_DESIGN,
        "version": ValueDimensionType.API_DESIGN,
    }
    
    def __init__(self):
        pass
    
    def generate(
        self,
        topics: List[TopicResult],
        comments: List[str],
        sentiments: List[SentimentResult]
    ) -> List[ValueDimension]:
        print("[ValueDimensionGenerator] 开始生成价值维度...")
        
        dimension_count = {}
        dimension_keywords = {}
        
        for topic in topics:
            for keyword in topic.keywords[:5]:
                for kw, dim_type in self.KEYWORD_TO_DIMENSION.items():
                    if kw in keyword:
                        dimension_count[dim_type] = dimension_count.get(dim_type, 0) + 1
                        if dim_type not in dimension_keywords:
                            dimension_keywords[dim_type] = []
                        if keyword not in dimension_keywords[dim_type]:
                            dimension_keywords[dim_type].append(keyword)
        
        dimensions = []
        dim_id = 0
        
        for dim_type, count in sorted(dimension_count.items(), key=lambda x: -x[1]):
            template = self.DIMENSION_TEMPLATES[dim_type]
            
            positive_examples = []
            negative_examples = []
            
            for comment, sentiment in zip(comments, sentiments):
                comment_lower = comment.lower()
                matches_keyword = any(kw in comment_lower for kw in dimension_keywords[dim_type])
                
                if matches_keyword:
                    if sentiment.label.value == "positive":
                        if len(positive_examples) < 3:
                            positive_examples.append(comment[:60] + "..." if len(comment) > 60 else comment)
                    elif sentiment.label.value == "negative":
                        if len(negative_examples) < 3:
                            negative_examples.append(comment[:60] + "..." if len(comment) > 60 else comment)
            
            importance = count / len(topics)
            sentiment_balance = self._calculate_sentiment_balance(positive_examples, negative_examples)
            
            dimensions.append(ValueDimension(
                id=dim_id,
                name=template["name"],
                type=dim_type,
                description=template["description"],
                keywords=dimension_keywords[dim_type],
                positive_examples=positive_examples,
                negative_examples=negative_examples,
                importance_score=importance,
                sentiment_balance=sentiment_balance
            ))
            dim_id += 1
        
        dimensions.sort(key=lambda d: -d.importance_score)
        
        print(f"[ValueDimensionGenerator] 生成了 {len(dimensions)} 个价值维度")
        
        return dimensions
    
    def _calculate_sentiment_balance(self, positive: List[str], negative: List[str]) -> float:
        total = len(positive) + len(negative)
        if total == 0:
            return 0.0
        return (len(positive) - len(negative)) / total
    
    def get_dimension_by_keyword(self, keyword: str) -> Optional[ValueDimensionType]:
        for kw, dim_type in self.KEYWORD_TO_DIMENSION.items():
            if kw in keyword.lower():
                return dim_type
        return None