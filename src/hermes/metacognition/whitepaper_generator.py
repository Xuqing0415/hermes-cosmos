from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import os

from .value_dimension_generator import ValueDimension
from .sentiment_analyzer import SentimentResult
from .topic_extractor import TopicResult


@dataclass
class WhitepaperConfig:
    output_dir: str = "./whitepapers"
    title: str = "A New Theory of Software Value: Derived from Community Voices"
    author: str = "AutoTestGen Metacognition Engine"
    include_introduction: bool = True
    include_methodology: bool = True
    include_dimensions: bool = True
    include_conclusion: bool = True
    include_recommendations: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "output_dir": self.output_dir,
            "title": self.title,
            "author": self.author,
            "include_introduction": self.include_introduction,
            "include_methodology": self.include_methodology,
            "include_dimensions": self.include_dimensions,
            "include_conclusion": self.include_conclusion,
            "include_recommendations": self.include_recommendations
        }


@dataclass
class WhitepaperReport:
    title: str = ""
    generated_at: str = ""
    author: str = ""
    dimensions: List[ValueDimension] = field(default_factory=list)
    sentiment_stats: Dict[str, Any] = field(default_factory=dict)
    topics: List[TopicResult] = field(default_factory=list)
    total_comments: int = 0
    markdown_content: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "generated_at": self.generated_at,
            "author": self.author,
            "dimensions": [d.to_dict() for d in self.dimensions],
            "sentiment_stats": self.sentiment_stats,
            "topics": [t.to_dict() for t in self.topics],
            "total_comments": self.total_comments
        }


class WhitepaperGenerator:
    INTRODUCTION_TEMPLATE = """# {title}

**Author**: {author}  
**Generated**: {date}  
**Data Source**: {total_comments} GitHub Issue comments from AutoTestGen community

## Abstract

This whitepaper presents a novel framework for understanding software quality derived from analyzing real user feedback. By applying natural language processing techniques to GitHub Issue comments, we identify latent dimensions of software value that go beyond traditional metrics like speed and correctness.

## Introduction

Software quality is often measured through technical metrics: performance benchmarks, test coverage, code complexity. However, these metrics fail to capture the **user experience** of quality - how users perceive, interact with, and value the software.

We analyzed {total_comments} user comments from the AutoTestGen project's GitHub Issues. Beyond traditional metrics, we discovered {n_dimensions} latent dimensions that significantly impact user satisfaction and system value:

{dimension_list}

This paper introduces these dimensions and proposes a new multi-objective optimization framework that includes them alongside traditional quality metrics.
"""
    
    METHODOLOGY_TEMPLATE = """## Methodology

### Data Collection

We collected {total_comments} user comments from GitHub Issues and Pull Requests. The comments cover various aspects of the AutoTestGen system including performance, documentation, debugging, API design, and reliability.

### Text Processing

1. **Cleaning**: Removed punctuation, numbers, and standard stopwords
2. **Tokenization**: Split text into words and bigrams
3. **TF-IDF Vectorization**: Transformed text into numerical features weighted by importance

### Topic Extraction

We used Non-negative Matrix Factorization (NMF) to identify {n_topics} latent topics from the comment corpus. Each topic is represented by a set of weighted keywords.

### Sentiment Analysis

Sentiment analysis was performed using VADER (Valence Aware Dictionary and sEntiment Reasoner) to classify each comment as positive, negative, or neutral, along with a confidence score.

### Value Dimension Generation

Topics were mapped to value dimensions through keyword matching and semantic analysis. Each dimension is characterized by:
- **Importance Score**: Based on topic prevalence
- **Sentiment Balance**: Ratio of positive to negative comments
- **Example Comments**: Representative positive and negative feedback
"""
    
    DIMENSION_TEMPLATE = """## Value Dimensions

### {dimension_number}. **{name}**

**Category**: {category}  
**Importance Score**: {importance:.2f}  
**Sentiment Balance**: {sentiment_balance:+.2f} ({sentiment_label})

{description}

#### Key Keywords
{keywords}

#### Positive Examples
{positive_examples}

#### Negative Examples
{negative_examples}

#### Discussion

{discussion}
"""
    
    CONCLUSION_TEMPLATE = """## Conclusion

Based on our analysis of {total_comments} user comments, we have identified {n_dimensions} latent dimensions of software value that complement traditional metrics:

{dimension_summary}

These dimensions represent **user-centric quality attributes** that significantly impact adoption, satisfaction, and long-term system success. We propose that software development and optimization processes should explicitly incorporate these dimensions alongside traditional technical metrics.

## Recommendations

1. **Integrate User Feedback into Development**: Regularly analyze GitHub Issues and community feedback to identify emerging value dimensions.

2. **Prioritize High-Impact Dimensions**: Focus resources on dimensions with high importance scores and negative sentiment balance.

3. **Design for Latent Dimensions**: Consider {dimension_names} when designing new features and improvements.

4. **Measure What Matters**: Expand quality metrics beyond technical benchmarks to include user-perceived dimensions.

## Future Work

- Extend analysis to include commit messages and pull request descriptions
- Incorporate temporal analysis to track dimension evolution over time
- Develop quantitative metrics for each value dimension
- Integrate with existing CI/CD pipelines for continuous value monitoring

---

*Generated by AutoTestGen Metacognition Engine*
"""
    
    def __init__(self, config: Optional[WhitepaperConfig] = None):
        self.config = config or WhitepaperConfig()
        
        if not os.path.exists(self.config.output_dir):
            os.makedirs(self.config.output_dir)
    
    def generate(
        self,
        dimensions: List[ValueDimension],
        sentiment_stats: Dict[str, Any],
        topics: List[TopicResult],
        total_comments: int
    ) -> WhitepaperReport:
        print("[WhitepaperGenerator] 开始生成白皮书...")
        
        parts = []
        
        if self.config.include_introduction:
            parts.append(self._generate_introduction(dimensions, total_comments))
        
        if self.config.include_methodology:
            parts.append(self._generate_methodology(total_comments, len(topics)))
        
        if self.config.include_dimensions:
            parts.append(self._generate_dimensions(dimensions))
        
        if self.config.include_conclusion:
            parts.append(self._generate_conclusion(dimensions, total_comments))
        
        if self.config.include_recommendations:
            pass
        
        markdown_content = "\n".join(parts)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"whitepaper_{timestamp}.md"
        filepath = os.path.join(self.config.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        latest_path = os.path.join(self.config.output_dir, "whitepaper.md")
        with open(latest_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        report = WhitepaperReport(
            title=self.config.title,
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            author=self.config.author,
            dimensions=dimensions,
            sentiment_stats=sentiment_stats,
            topics=topics,
            total_comments=total_comments,
            markdown_content=markdown_content
        )
        
        print(f"[WhitepaperGenerator] 白皮书已保存: {filepath}")
        
        return report
    
    def _generate_introduction(self, dimensions: List[ValueDimension], total_comments: int) -> str:
        dimension_list = "\n".join([
            f"- **{d.name}**: {d.description[:60]}..."
            for d in dimensions[:5]
        ])
        
        return self.INTRODUCTION_TEMPLATE.format(
            title=self.config.title,
            author=self.config.author,
            date=datetime.now().strftime("%Y-%m-%d"),
            total_comments=total_comments,
            n_dimensions=len(dimensions),
            dimension_list=dimension_list
        )
    
    def _generate_methodology(self, total_comments: int, n_topics: int) -> str:
        return self.METHODOLOGY_TEMPLATE.format(
            total_comments=total_comments,
            n_topics=n_topics
        )
    
    def _generate_dimensions(self, dimensions: List[ValueDimension]) -> str:
        parts = []
        
        for i, dim in enumerate(dimensions, 1):
            keywords = ", ".join(dim.keywords[:5])
            
            positive_examples = "\n".join([
                f"- {ex}" for ex in dim.positive_examples
            ]) if dim.positive_examples else "- No positive examples found"
            
            negative_examples = "\n".join([
                f"- {ex}" for ex in dim.negative_examples
            ]) if dim.negative_examples else "- No negative examples found"
            
            sentiment_label = "Mostly Positive" if dim.sentiment_balance > 0.2 else \
                             "Mostly Negative" if dim.sentiment_balance < -0.2 else "Balanced"
            
            discussion = self._generate_discussion(dim)
            
            parts.append(self.DIMENSION_TEMPLATE.format(
                dimension_number=i,
                name=dim.name,
                category=dim.type.value,
                importance=dim.importance_score,
                sentiment_balance=dim.sentiment_balance,
                sentiment_label=sentiment_label,
                description=dim.description,
                keywords=keywords,
                positive_examples=positive_examples,
                negative_examples=negative_examples,
                discussion=discussion
            ))
        
        return "\n".join(parts)
    
    def _generate_discussion(self, dim: ValueDimension) -> str:
        discussions = {
            "Gently Degradable": "Users value systems that degrade gracefully under stress rather than failing abruptly. The ability to maintain partial functionality during resource constraints is a key differentiator for production-quality software.",
            "Explainable Failures": "When tests fail, users need actionable diagnostic information. Cryptic error messages lead to frustration and increased debugging time. Clear, contextual error messages significantly improve the developer experience.",
            "Convergent Learning": "The system's ability to adapt to user patterns over time creates a virtuous cycle of improvement. Users perceive the system as 'learning' and become more engaged and satisfied.",
            "Performance": "While performance is a traditional metric, user feedback emphasizes the **perception** of speed. Even moderate improvements in response time can have a significant impact on user satisfaction.",
            "Usability": "Intuitive interfaces reduce onboarding time and increase adoption. Confusing APIs and CLIs are a common source of frustration and abandonment.",
            "Reliability": "Consistent performance builds trust. Flaky tests and random crashes erode confidence in the system's ability to deliver reliable results.",
            "Documentation Quality": "High-quality documentation is essential for adoption. Sparse or unclear documentation is a major barrier to entry for new users.",
            "API Design": "Clean, consistent APIs reduce integration effort and increase developer productivity. Breaking changes without proper versioning damage trust and require significant user effort to adapt.",
        }
        
        return discussions.get(dim.name, "This dimension represents an important aspect of user-perceived software quality. Further research is needed to understand its full implications.")
    
    def _generate_conclusion(self, dimensions: List[ValueDimension], total_comments: int) -> str:
        dimension_summary = "\n".join([
            f"- **{d.name}**: {d.description[:80]}..."
            for d in dimensions
        ])
        
        dimension_names = ", ".join([d.name for d in dimensions[:3]])
        
        return self.CONCLUSION_TEMPLATE.format(
            total_comments=total_comments,
            n_dimensions=len(dimensions),
            dimension_summary=dimension_summary,
            dimension_names=dimension_names
        )