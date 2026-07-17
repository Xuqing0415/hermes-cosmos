from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
import math

from .abstract_pattern_extractor import AbstractPattern, AbstractPatternType


@dataclass
class SimilarityMatch:
    pattern_a: AbstractPattern
    pattern_b: AbstractPattern
    similarity: float
    shared_keywords: List[str]
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_a_id": self.pattern_a.id,
            "pattern_a_domain": self.pattern_a.domain,
            "pattern_b_id": self.pattern_b.id,
            "pattern_b_domain": self.pattern_b.domain,
            "similarity": round(self.similarity, 2),
            "shared_keywords": self.shared_keywords,
            "explanation": self.explanation
        }


@dataclass
class PatternCluster:
    abstract_type: AbstractPatternType
    patterns: List[AbstractPattern] = field(default_factory=list)
    representative_keywords: List[str] = field(default_factory=list)
    cluster_similarity: float = 0.0

    @property
    def domains(self) -> List[str]:
        return list(set(p.domain for p in self.patterns))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "abstract_type": self.abstract_type.value,
            "pattern_count": len(self.patterns),
            "domains": self.domains,
            "representative_keywords": self.representative_keywords,
            "cluster_similarity": round(self.cluster_similarity, 2),
            "patterns": [p.to_dict() for p in self.patterns]
        }


PATTERN_TYPE_SIMILARITY: Dict[str, Dict[str, float]] = {
    AbstractPatternType.BOUNDARY_CHECK_MISSING.value: {
        AbstractPatternType.BOUNDARY_CHECK_MISSING.value: 1.0,
        AbstractPatternType.RESOURCE_LIMIT_MISSING.value: 0.75,
        AbstractPatternType.INPUT_VALIDATION_MISSING.value: 0.6,
        AbstractPatternType.DIVISION_BY_ZERO.value: 0.5,
        AbstractPatternType.UNINITIALIZED_VARIABLE.value: 0.45,
        AbstractPatternType.API_VERSION_DEPRECATED.value: 0.2,
        AbstractPatternType.PERFORMANCE_BOTTLENECK.value: 0.15,
    },
    AbstractPatternType.RESOURCE_LIMIT_MISSING.value: {
        AbstractPatternType.BOUNDARY_CHECK_MISSING.value: 0.75,
        AbstractPatternType.RESOURCE_LIMIT_MISSING.value: 1.0,
        AbstractPatternType.INPUT_VALIDATION_MISSING.value: 0.5,
        AbstractPatternType.DIVISION_BY_ZERO.value: 0.3,
        AbstractPatternType.UNINITIALIZED_VARIABLE.value: 0.35,
        AbstractPatternType.API_VERSION_DEPRECATED.value: 0.25,
        AbstractPatternType.PERFORMANCE_BOTTLENECK.value: 0.4,
    },
    AbstractPatternType.INPUT_VALIDATION_MISSING.value: {
        AbstractPatternType.BOUNDARY_CHECK_MISSING.value: 0.6,
        AbstractPatternType.RESOURCE_LIMIT_MISSING.value: 0.5,
        AbstractPatternType.INPUT_VALIDATION_MISSING.value: 1.0,
        AbstractPatternType.DIVISION_BY_ZERO.value: 0.4,
        AbstractPatternType.UNINITIALIZED_VARIABLE.value: 0.55,
        AbstractPatternType.API_VERSION_DEPRECATED.value: 0.2,
        AbstractPatternType.PERFORMANCE_BOTTLENECK.value: 0.3,
    },
}


DOMAIN_WEIGHTS: Dict[str, float] = {
    "mlir": 1.0,
    "k8s": 1.0,
    "default": 1.0,
}


class PatternSimilarityEngine:
    def __init__(self):
        self._patterns: List[AbstractPattern] = []

    def add_patterns(self, patterns: List[AbstractPattern]):
        self._patterns.extend(patterns)

    def clear_patterns(self):
        self._patterns.clear()

    def calculate_similarity(self, pattern_a: AbstractPattern, pattern_b: AbstractPattern) -> float:
        if pattern_a.id == pattern_b.id:
            return 1.0

        type_sim = self._calculate_type_similarity(pattern_a, pattern_b)
        keyword_sim = self._calculate_keyword_similarity(pattern_a, pattern_b)
        message_sim = self._calculate_message_similarity(pattern_a, pattern_b)
        domain_penalty = self._calculate_domain_penalty(pattern_a, pattern_b)

        total = (
            0.5 * type_sim +
            0.3 * keyword_sim +
            0.15 * message_sim +
            0.05 * domain_penalty
        )

        return max(0.0, min(1.0, total))

    def find_similar_patterns(self, target: AbstractPattern, threshold: float = 0.5) -> List[SimilarityMatch]:
        matches = []
        for pattern in self._patterns:
            if pattern.id == target.id:
                continue

            similarity = self.calculate_similarity(target, pattern)
            if similarity >= threshold:
                shared_keywords = list(set(target.keywords) & set(pattern.keywords))
                explanation = self._generate_explanation(target, pattern, similarity, shared_keywords)

                matches.append(SimilarityMatch(
                    pattern_a=target,
                    pattern_b=pattern,
                    similarity=similarity,
                    shared_keywords=shared_keywords,
                    explanation=explanation
                ))

        matches.sort(key=lambda m: m.similarity, reverse=True)
        return matches

    def find_cross_domain_similarities(self, threshold: float = 0.5) -> List[SimilarityMatch]:
        matches = []
        for i, pattern_a in enumerate(self._patterns):
            for j, pattern_b in enumerate(self._patterns):
                if i >= j or pattern_a.domain == pattern_b.domain:
                    continue

                similarity = self.calculate_similarity(pattern_a, pattern_b)
                if similarity >= threshold:
                    shared_keywords = list(set(pattern_a.keywords) & set(pattern_b.keywords))
                    explanation = self._generate_explanation(pattern_a, pattern_b, similarity, shared_keywords)

                    matches.append(SimilarityMatch(
                        pattern_a=pattern_a,
                        pattern_b=pattern_b,
                        similarity=similarity,
                        shared_keywords=shared_keywords,
                        explanation=explanation
                    ))

        matches.sort(key=lambda m: m.similarity, reverse=True)
        return matches

    def cluster_patterns(self, threshold: float = 0.5) -> List[PatternCluster]:
        clusters: Dict[str, PatternCluster] = {}

        for pattern in self._patterns:
            type_key = pattern.pattern_type.value

            if type_key not in clusters:
                clusters[type_key] = PatternCluster(
                    abstract_type=pattern.pattern_type,
                    patterns=[],
                    representative_keywords=[],
                    cluster_similarity=0.0
                )

            clusters[type_key].patterns.append(pattern)

        for cluster in clusters.values():
            if len(cluster.patterns) >= 2:
                similarities = []
                for i, p1 in enumerate(cluster.patterns):
                    for j, p2 in enumerate(cluster.patterns):
                        if i < j:
                            similarities.append(self.calculate_similarity(p1, p2))

                cluster.cluster_similarity = sum(similarities) / len(similarities) if similarities else 0.0

            all_keywords = []
            for p in cluster.patterns:
                all_keywords.extend(p.keywords)

            keyword_counts = {}
            for kw in all_keywords:
                keyword_counts[kw] = keyword_counts.get(kw, 0) + 1

            sorted_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)
            cluster.representative_keywords = [kw for kw, _ in sorted_keywords[:5]]

        return sorted(clusters.values(), key=lambda c: len(c.patterns), reverse=True)

    def _calculate_type_similarity(self, pattern_a: AbstractPattern, pattern_b: AbstractPattern) -> float:
        type_a = pattern_a.pattern_type.value
        type_b = pattern_b.pattern_type.value

        if type_a == type_b:
            return 1.0

        if type_a in PATTERN_TYPE_SIMILARITY:
            return PATTERN_TYPE_SIMILARITY[type_a].get(type_b, 0.2)

        return 0.2

    def _calculate_keyword_similarity(self, pattern_a: AbstractPattern, pattern_b: AbstractPattern) -> float:
        if not pattern_a.keywords or not pattern_b.keywords:
            return 0.0

        intersection = set(pattern_a.keywords) & set(pattern_b.keywords)
        union = set(pattern_a.keywords) | set(pattern_b.keywords)

        if not union:
            return 0.0

        return len(intersection) / len(union)

    def _calculate_message_similarity(self, pattern_a: AbstractPattern, pattern_b: AbstractPattern) -> float:
        if not pattern_a.message or not pattern_b.message:
            return 0.0

        words_a = set(pattern_a.message.lower().split())
        words_b = set(pattern_b.message.lower().split())

        if not words_a or not words_b:
            return 0.0

        intersection = words_a & words_b
        union = words_a | words_b

        if not union:
            return 0.0

        return len(intersection) / len(union)

    def _calculate_domain_penalty(self, pattern_a: AbstractPattern, pattern_b: AbstractPattern) -> float:
        if pattern_a.domain == pattern_b.domain:
            return 1.0

        weight_a = DOMAIN_WEIGHTS.get(pattern_a.domain, 1.0)
        weight_b = DOMAIN_WEIGHTS.get(pattern_b.domain, 1.0)

        return 0.8 + (weight_a + weight_b) * 0.1

    def _generate_explanation(self, pattern_a: AbstractPattern, pattern_b: AbstractPattern,
                              similarity: float, shared_keywords: List[str]) -> str:
        if pattern_a.pattern_type == pattern_b.pattern_type:
            type_desc = "same abstract pattern type"
        else:
            type_desc = f"related pattern types ({pattern_a.pattern_type.value} ↔ {pattern_b.pattern_type.value})"

        if shared_keywords:
            keyword_desc = f"shared keywords: {', '.join(shared_keywords)}"
        else:
            keyword_desc = "no shared keywords"

        return f"Cross-domain similarity ({similarity:.2f}): {type_desc}, {keyword_desc}"