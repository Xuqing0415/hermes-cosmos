from typing import List, Dict, Optional, Any
from .types import GrammarRule, GrammarSpec, GrammarFeedback, RuleType, GrammarStatus
import time

class GrammarEvolver:
    """Evolves AutoLang grammar based on usage feedback and test pass rates."""
    
    def __init__(self):
        self.grammar_history: List[GrammarSpec] = []
        self.feedback_history: List[GrammarFeedback] = []
        self.current_grammar: GrammarSpec = self._initialize_default_grammar()
    
    def _initialize_default_grammar(self) -> GrammarSpec:
        rules = [
            GrammarRule("rule_001", "test_case", RuleType.SYNTAX, 
                       "TEST <function> WITH <input> EXPECT <output>",
                       "Generate a test case for a function"),
            GrammarRule("rule_002", "fix_directive", RuleType.SYNTAX,
                       "FIX <error_type> IN <function> USING <strategy>",
                       "Generate a fix for an error"),
            GrammarRule("rule_003", "proof_obligation", RuleType.SYNTAX,
                       "PROVE <property> FOR <function> ASSUMING <preconditions>",
                       "Generate a proof obligation"),
        ]
        spec = GrammarSpec("AutoLang", "1.0", rules, "Default AutoLang grammar specification")
        self.grammar_history.append(spec)
        return spec
    
    def evolve_grammar(self, feedbacks: List[GrammarFeedback]) -> GrammarSpec:
        """Evolve the grammar based on feedback. Returns a new GrammarSpec."""
        new_rules = []
        for rule in self.current_grammar.rules:
            matching_feedback = [f for f in feedbacks if f.rule_id == rule.rule_id]
            if matching_feedback:
                fb = matching_feedback[0]
                updated_rule = GrammarRule(
                    rule_id=rule.rule_id,
                    name=rule.name,
                    rule_type=rule.rule_type,
                    pattern=rule.pattern,
                    description=rule.description,
                    status=GrammarStatus.DEPRECATED if fb.test_pass_rate < 0.3 else GrammarStatus.STABLE,
                    usage_count=fb.usage_count,
                    success_rate=fb.test_pass_rate,
                    created_at=rule.created_at,
                    version=self._bump_version(rule.version)
                )
                new_rules.append(updated_rule)
                
                if fb.test_pass_rate < 0.5 and fb.suggestion:
                    new_rule = self._generate_improved_rule(rule, fb)
                    new_rules.append(new_rule)
            else:
                new_rules.append(rule)
        
        new_version = self._bump_version(self.current_grammar.version)
        new_spec = GrammarSpec(
            name=self.current_grammar.name,
            version=new_version,
            rules=new_rules,
            description=self.current_grammar.description
        )
        
        self.grammar_history.append(new_spec)
        self.current_grammar = new_spec
        self.feedback_history.extend(feedbacks)
        
        return new_spec
    
    def _generate_improved_rule(self, original: GrammarRule, feedback: GrammarFeedback) -> GrammarRule:
        """Generate an improved version of a rule based on feedback."""
        return GrammarRule(
            rule_id=f"{original.rule_id}_v2",
            name=original.name,
            rule_type=original.rule_type,
            pattern=feedback.suggestion if feedback.suggestion else original.pattern + " [IMPROVED]",
            description=f"Improved version of {original.name}. Issues: {'; '.join(feedback.error_patterns)}",
            status=GrammarStatus.EXPERIMENTAL,
            usage_count=0,
            success_rate=0.0,
            version="1.0"
        )
    
    def add_rule(self, name: str, rule_type: RuleType, pattern: str, description: str = "") -> GrammarRule:
        rule_id = f"rule_{len(self.current_grammar.rules) + 1:03d}"
        rule = GrammarRule(rule_id, name, rule_type, pattern, description, GrammarStatus.EXPERIMENTAL)
        self.current_grammar.rules.append(rule)
        return rule
    
    def get_rules_by_type(self, rule_type: RuleType) -> List[GrammarRule]:
        return [r for r in self.current_grammar.rules if r.rule_type == rule_type]
    
    def get_stable_rules(self) -> List[GrammarRule]:
        return [r for r in self.current_grammar.rules if r.status == GrammarStatus.STABLE]
    
    def _bump_version(self, version: str) -> str:
        parts = version.split(".")
        if len(parts) == 2:
            major, minor = int(parts[0]), int(parts[1])
            return f"{major}.{minor + 1}"
        return "1.1"
    
    def get_grammar_summary(self) -> Dict[str, Any]:
        return {
            "name": self.current_grammar.name,
            "version": self.current_grammar.version,
            "total_rules": len(self.current_grammar.rules),
            "stable_rules": len(self.get_stable_rules()),
            "history_count": len(self.grammar_history),
            "feedback_count": len(self.feedback_history)
        }
