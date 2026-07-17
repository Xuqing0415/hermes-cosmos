from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum


class SentimentLabel(Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


@dataclass
class SentimentResult:
    score: float
    label: SentimentLabel
    confidence: float
    positive_words: List[str] = field(default_factory=list)
    negative_words: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": self.score,
            "label": self.label.value,
            "confidence": self.confidence,
            "positive_words": self.positive_words,
            "negative_words": self.negative_words
        }


class SentimentAnalyzer:
    POSITIVE_KEYWORDS = {
        "good", "great", "excellent", "amazing", "love", "best", "awesome",
        "nice", "graceful", "helpful", "improved", "intuitive", "clean",
        "consistent", "easy", "painless", "impressive", "solid", "smart",
        "adaptive", "learned", "perfect", "wonderful", "fantastic", "beautiful",
        "successful", "working", "stable", "reliable", "fast", "quick", "smooth"
    }
    
    NEGATIVE_KEYWORDS = {
        "slow", "crash", "freeze", "fail", "failed", "failing", "error", "errors",
        "bug", "bugs", "broken", "missing", "lacking", "confusing", "cryptic",
        "frustrating", "painful", "terrible", "bad", "awful", "worse", "problem",
        "problems", "issue", "issues", "flaky", "unstable", "memory", "leak",
        "timeout", "paused", "stopped", "abruptly", "silently", "degrade", "degraded"
    }
    
    def __init__(self):
        self._use_vader = self._try_import_vader()
    
    def _try_import_vader(self) -> bool:
        try:
            import nltk
            nltk.download('vader_lexicon', quiet=True)
            from nltk.sentiment import SentimentIntensityAnalyzer
            self._vader = SentimentIntensityAnalyzer()
            return True
        except ImportError:
            print("[SentimentAnalyzer] NLTK/VADER 未安装，使用关键词统计方法")
            return False
    
    def analyze(self, text: str) -> SentimentResult:
        if self._use_vader:
            return self._analyze_with_vader(text)
        else:
            return self._analyze_with_keywords(text)
    
    def analyze_batch(self, texts: List[str]) -> List[SentimentResult]:
        return [self.analyze(text) for text in texts]
    
    def _analyze_with_vader(self, text: str) -> SentimentResult:
        scores = self._vader.polarity_scores(text)
        compound = scores['compound']
        
        if compound >= 0.05:
            label = SentimentLabel.POSITIVE
        elif compound <= -0.05:
            label = SentimentLabel.NEGATIVE
        else:
            label = SentimentLabel.NEUTRAL
        
        confidence = abs(compound)
        
        positive_words = []
        negative_words = []
        
        for word in text.lower().split():
            if word in self.POSITIVE_KEYWORDS:
                positive_words.append(word)
            elif word in self.NEGATIVE_KEYWORDS:
                negative_words.append(word)
        
        return SentimentResult(
            score=compound,
            label=label,
            confidence=confidence,
            positive_words=positive_words,
            negative_words=negative_words
        )
    
    def _analyze_with_keywords(self, text: str) -> SentimentResult:
        words = text.lower().split()
        
        pos_count = sum(1 for w in words if w in self.POSITIVE_KEYWORDS)
        neg_count = sum(1 for w in words if w in self.NEGATIVE_KEYWORDS)
        
        total = pos_count + neg_count
        
        if total == 0:
            score = 0.0
            label = SentimentLabel.NEUTRAL
            confidence = 0.0
        else:
            score = (pos_count - neg_count) / total
            if score > 0.2:
                label = SentimentLabel.POSITIVE
            elif score < -0.2:
                label = SentimentLabel.NEGATIVE
            else:
                label = SentimentLabel.NEUTRAL
            confidence = min(total / 5, 1.0)
        
        positive_words = [w for w in words if w in self.POSITIVE_KEYWORDS]
        negative_words = [w for w in words if w in self.NEGATIVE_KEYWORDS]
        
        return SentimentResult(
            score=score,
            label=label,
            confidence=confidence,
            positive_words=positive_words,
            negative_words=negative_words
        )
    
    def get_sentiment_stats(self, results: List[SentimentResult]) -> Dict[str, Any]:
        total = len(results)
        positive = sum(1 for r in results if r.label == SentimentLabel.POSITIVE)
        negative = sum(1 for r in results if r.label == SentimentLabel.NEGATIVE)
        neutral = sum(1 for r in results if r.label == SentimentLabel.NEUTRAL)
        
        avg_score = sum(r.score for r in results) / total if total > 0 else 0
        avg_confidence = sum(r.confidence for r in results) / total if total > 0 else 0
        
        return {
            "total": total,
            "positive": positive,
            "negative": negative,
            "neutral": neutral,
            "positive_ratio": positive / total if total > 0 else 0,
            "negative_ratio": negative / total if total > 0 else 0,
            "avg_score": avg_score,
            "avg_confidence": avg_confidence
        }