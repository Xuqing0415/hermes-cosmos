from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
import re
import string

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF


@dataclass
class TopicResult:
    topic_id: int
    name: str
    keywords: List[str]
    weights: List[float]
    top_documents: List[str] = field(default_factory=list)
    relevance_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic_id": self.topic_id,
            "name": self.name,
            "keywords": self.keywords,
            "weights": [round(w, 4) for w in self.weights],
            "top_documents": self.top_documents,
            "relevance_score": round(self.relevance_score, 4)
        }


class TopicExtractor:
    STOP_WORDS = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "must", "shall", "can", "need", "dare",
        "ought", "used", "to", "of", "in", "for", "on", "with", "at", "by",
        "from", "as", "into", "through", "during", "before", "after", "above",
        "below", "between", "under", "again", "further", "then", "once", "here",
        "there", "when", "where", "why", "how", "all", "each", "few", "more",
        "most", "other", "some", "such", "no", "nor", "not", "only", "own",
        "same", "so", "than", "too", "very", "just", "but", "if", "or", "and",
        "because", "until", "while", "this", "that", "these", "those", "what",
        "which", "who", "whom", "i", "you", "he", "she", "it", "we", "they",
        "me", "him", "her", "us", "them", "my", "your", "his", "its", "our",
        "their", "yourself", "himself", "herself", "itself", "ourselves",
        "themselves", "myself", "about", "against", "amid", "among", "around",
        "beside", "beyond", "concerning", "considering", "despite", "except",
        "inside", "outside", "over", "past", "regarding", "round", "since",
        "throughout", "toward", "towards", "underneath", "unlike", "until",
        "upon", "within", "without", "wouldn", "couldn", "shouldn", "mightn",
        "mustn", "needn", "daren", "oughtn", "usedn", "wasn", "weren", "hasn",
        "haven", "hadn", "didn", "don", "doesn", "isn", "aren", "ain", "won",
        "can't", "couldn't", "wouldn't", "shouldn't", "mightn't", "mustn't",
        "needn't", "daren't", "oughtn't", "usedn't", "wasn't", "weren't",
        "hasn't", "haven't", "hadn't", "didn't", "don't", "doesn't", "isn't",
        "aren't", "ain't", "won't", "system", "test", "tests", "autotestgen",
        "code", "software", "issue", "issues", "bug", "bugs", "feature",
        "features", "fix", "fixes", "error", "errors", "problem", "problems"
    }
    
    TOPIC_NAME_TEMPLATES = {
        "performance": ["Performance", "Speed", "Efficiency"],
        "usability": ["Usability", "User Experience", "Ease of Use"],
        "documentation": ["Documentation", "Docs", "Guide"],
        "debugging": ["Debugging", "Error Handling", "Troubleshooting"],
        "api": ["API Design", "Interface", "SDK"],
        "reliability": ["Reliability", "Stability", "Resilience"],
        "learning": ["Learning", "Adaptation", "Intelligence"],
        "memory": ["Memory", "Resource Usage", "Scalability"],
        "error": ["Error Handling", "Failure Recovery", "Graceful Degradation"],
        "crash": ["Stability", "Crash Prevention", "Reliability"],
        "slow": ["Performance", "Speed Optimization", "Response Time"],
        "fast": ["Performance", "Speed", "Responsiveness"],
        "confusing": ["Usability", "Clarity", "Intuitiveness"],
        "helpful": ["Usability", "Helpfulness", "User Support"],
        "intuitive": ["Usability", "Intuitiveness", "Simplicity"],
        "graceful": ["Resilience", "Graceful Degradation", "Failure Handling"],
        "resilience": ["Resilience", "Fault Tolerance", "Recovery"],
        "adaptive": ["Adaptation", "Learning", "Evolution"],
        "learned": ["Learning", "Adaptation", "Intelligence"],
        "stable": ["Stability", "Reliability", "Consistency"],
        "consistent": ["Consistency", "API Design", "Reliability"],
    }
    
    def __init__(self, n_topics: int = 5, max_features: int = 500):
        self.n_topics = n_topics
        self.max_features = max_features
        self.vectorizer = None
        self.nmf = None
    
    def extract(self, documents: List[str]) -> List[TopicResult]:
        print(f"[TopicExtractor] 开始提取 {self.n_topics} 个主题...")
        
        cleaned_docs = [self._clean_text(doc) for doc in documents]
        
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            stop_words=list(self.STOP_WORDS),
            ngram_range=(1, 2),
            lowercase=True
        )
        
        tfidf_matrix = self.vectorizer.fit_transform(cleaned_docs)
        
        self.nmf = NMF(
            n_components=self.n_topics,
            random_state=42,
            max_iter=1000,
            tol=1e-4
        )
        
        topic_matrix = self.nmf.fit_transform(tfidf_matrix)
        
        feature_names = self.vectorizer.get_feature_names_out()
        
        topics = []
        for topic_idx in range(self.n_topics):
            topic_weights = self.nmf.components_[topic_idx]
            top_indices = topic_weights.argsort()[::-1][:10]
            keywords = [feature_names[i] for i in top_indices]
            weights = [topic_weights[i] for i in top_indices]
            
            name = self._generate_topic_name(keywords)
            
            top_docs = self._get_top_documents(topic_matrix, documents, topic_idx, n=3)
            
            relevance = self._calculate_relevance(topic_matrix, topic_idx)
            
            topics.append(TopicResult(
                topic_id=topic_idx,
                name=name,
                keywords=keywords,
                weights=weights,
                top_documents=top_docs,
                relevance_score=relevance
            ))
        
        topics.sort(key=lambda t: -t.relevance_score)
        
        print(f"[TopicExtractor] 提取了 {len(topics)} 个主题")
        
        return topics
    
    def _clean_text(self, text: str) -> str:
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def _generate_topic_name(self, keywords: List[str]) -> str:
        for keyword in keywords[:5]:
            for template_key, names in self.TOPIC_NAME_TEMPLATES.items():
                if template_key in keyword:
                    return names[0]
        
        return keywords[0].capitalize() if keywords else "Uncategorized"
    
    def _get_top_documents(self, topic_matrix, documents, topic_idx, n: int = 3) -> List[str]:
        doc_scores = topic_matrix[:, topic_idx]
        top_indices = doc_scores.argsort()[::-1][:n]
        return [documents[i][:80] + "..." if len(documents[i]) > 80 else documents[i] 
                for i in top_indices]
    
    def _calculate_relevance(self, topic_matrix, topic_idx) -> float:
        scores = topic_matrix[:, topic_idx]
        avg_score = scores.mean()
        max_score = scores.max()
        return (avg_score * 0.5) + (max_score * 0.5)
    
    def get_document_topics(self, document: str) -> List[Dict[str, Any]]:
        if self.vectorizer is None or self.nmf is None:
            return []
        
        cleaned = self._clean_text(document)
        tfidf = self.vectorizer.transform([cleaned])
        topic_scores = self.nmf.transform(tfidf)[0]
        
        return [
            {"topic_id": i, "score": float(score), "name": f"Topic_{i}"}
            for i, score in enumerate(topic_scores)
            if score > 0.01
        ]