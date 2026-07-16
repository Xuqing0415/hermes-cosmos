from typing import List, Dict, Optional, Any
from .types import MigrationRule, MigrationResult, Language, MigrationStatus
import re

class KnowledgeMigrator:
    """Migrates test/fix rules and knowledge across programming languages."""
    
    def __init__(self):
        self.migration_rules: Dict[str, List[MigrationRule]] = {}
        self.migration_history: List[MigrationResult] = []
        self._initialize_default_rules()
    
    def _initialize_default_rules(self):
        # Python to Rust
        self.add_rule(Language.PYTHON, Language.RUST,
                     r"assert\s+(\w+)\s*==\s*(.+)", "assert_eq!($1, $2);",
                     "Assert equality: Python to Rust")
        self.add_rule(Language.PYTHON, Language.RUST,
                     r"def\s+(\w+)\s*\(([^)]*)\):", "fn $1($2) {",
                     "Function definition: Python to Rust")
        
        # Python to Go
        self.add_rule(Language.PYTHON, Language.GO,
                     r"def\s+(\w+)\s*\(([^)]*)\):", "func $1($2) {",
                     "Function definition: Python to Go")
        self.add_rule(Language.PYTHON, Language.GO,
                     r"assert\s+(\w+)\s*==\s*(.+)", "if $1 != $2 { t.Errorf(\"mismatch\") }",
                     "Assert equality: Python to Go")
        
        # Python to Java
        self.add_rule(Language.PYTHON, Language.JAVA,
                     r"def\s+(\w+)\s*\(([^)]*)\):", "public static void $1($2) {",
                     "Function definition: Python to Java")
        self.add_rule(Language.PYTHON, Language.JAVA,
                     r"assert\s+(\w+)\s*==\s*(.+)", "assertEquals($1, $2);",
                     "Assert equality: Python to Java")
    
    def add_rule(self, source_lang: Language, target_lang: Language,
                 pattern: str, translation: str, description: str = "",
                 confidence: float = 1.0) -> MigrationRule:
        rule_id = f"{source_lang.value}_to_{target_lang.value}_{len(self._get_rules(source_lang, target_lang)) + 1:03d}"
        rule = MigrationRule(
            rule_id=rule_id,
            source_lang=source_lang,
            target_lang=target_lang,
            pattern=pattern,
            translation=translation,
            description=description,
            confidence=confidence
        )
        key = f"{source_lang.value}_{target_lang.value}"
        if key not in self.migration_rules:
            self.migration_rules[key] = []
        self.migration_rules[key].append(rule)
        return rule
    
    def _get_rules(self, source_lang: Language, target_lang: Language) -> List[MigrationRule]:
        key = f"{source_lang.value}_{target_lang.value}"
        return self.migration_rules.get(key, [])
    
    def migrate(self, source_code: str, source_lang: Language, 
                target_lang: Language) -> MigrationResult:
        rules = self._get_rules(source_lang, target_lang)
        if not rules:
            return MigrationResult(
                source_code=source_code,
                target_code=source_code,
                source_lang=source_lang,
                target_lang=target_lang,
                status=MigrationStatus.FAILED,
                warnings=[f"No migration rules from {source_lang.value} to {target_lang.value}"]
            )
        
        target_code = source_code
        rules_applied = []
        warnings = []
        
        for rule in rules:
            try:
                new_code = re.sub(rule.pattern, rule.translation, target_code)
                if new_code != target_code:
                    rules_applied.append(rule.rule_id)
                    rule.usage_count += 1
                    target_code = new_code
            except re.error as e:
                warnings.append(f"Rule {rule.rule_id} regex error: {e}")
        
        if not rules_applied:
            status = MigrationStatus.PARTIAL
            warnings.append("No rules were applicable to the source code")
        else:
            status = MigrationStatus.SUCCESS
        
        avg_confidence = sum(r.confidence for r in rules if r.rule_id in rules_applied) / len(rules_applied) if rules_applied else 0.0
        
        result = MigrationResult(
            source_code=source_code,
            target_code=target_code,
            source_lang=source_lang,
            target_lang=target_lang,
            status=status,
            rules_applied=rules_applied,
            warnings=warnings,
            confidence=avg_confidence
        )
        
        self.migration_history.append(result)
        return result
    
    def migrate_test_suite(self, tests: List[Dict[str, Any]], 
                          source_lang: Language, target_lang: Language) -> List[MigrationResult]:
        results = []
        for test in tests:
            code = test.get("code", "")
            result = self.migrate(code, source_lang, target_lang)
            results.append(result)
        return results
    
    def get_supported_pairs(self) -> List[tuple]:
        return [(k.split("_")[0], k.split("_")[1]) for k in self.migration_rules.keys()]
    
    def get_migration_summary(self) -> Dict[str, Any]:
        total_rules = sum(len(rules) for rules in self.migration_rules.values())
        return {
            "total_rules": total_rules,
            "supported_pairs": len(self.get_supported_pairs()),
            "total_migrations": len(self.migration_history),
            "successful_migrations": sum(1 for r in self.migration_history if r.status == MigrationStatus.SUCCESS)
        }
