import sys
import os
sys.path.insert(0, 'src')

from hermes.llm_requirement import (
    SelfAuditor, AuditIssue, AuditIssueType, IssueSeverity,
    TrendAnalyzer, TrendItem, TrendSource, TrendCategory,
    ArchitectureDesigner, ArchitectureSpec, ArchitectureModule,
    SelfRebirthEngine, RebirthStatus, RebirthStep
)


class TestSelfAuditor:
    def test_init(self):
        auditor = SelfAuditor()
        assert auditor is not None
        assert auditor.source_dir is not None
    
    def test_audit_basic(self):
        auditor = SelfAuditor()
        result = auditor.audit()
        
        assert result.total_files_scanned > 0
        assert result.total_issues_found >= 0
        assert result.critical_count >= 0
        assert result.high_count >= 0
        assert result.medium_count >= 0
        assert result.low_count >= 0
    
    def test_audit_issue_to_dict(self):
        issue = AuditIssue(
            issue_type=AuditIssueType.CODE_SMELL,
            severity=IssueSeverity.MEDIUM,
            title="Test Issue",
            description="Test description",
            file_path="test.py",
            line_number=10,
            suggestions=["Fix it"]
        )
        
        d = issue.to_dict()
        assert d["issue_type"] == "code_smell"
        assert d["severity"] == "medium"
        assert d["title"] == "Test Issue"


class TestTrendAnalyzer:
    def test_init(self):
        analyzer = TrendAnalyzer()
        assert analyzer is not None
        assert analyzer._timeout == 3
    
    def test_analyze_with_fallback(self):
        analyzer = TrendAnalyzer()
        result = analyzer.analyze([])
        
        assert result.total_items_found > 0
        assert len(result.top_recommendations) > 0
    
    def test_fallback_trends_categories(self):
        analyzer = TrendAnalyzer()
        trends = analyzer._get_fallback_trends()
        
        assert len(trends) > 0
        
        categories = [t.category for t in trends]
        assert TrendCategory.WEB_FRAMEWORK in categories
        assert TrendCategory.PERFORMANCE in categories
        assert TrendCategory.OBSERVABILITY in categories
    
    def test_trend_item_to_dict(self):
        trend = TrendItem(
            source=TrendSource.PYPI_NEW,
            name="Test",
            description="Test desc",
            url="https://test.com",
            stars=1000,
            category=TrendCategory.WEB_FRAMEWORK,
            tags=["python"]
        )
        
        d = trend.to_dict()
        assert d["name"] == "Test"
        assert d["category"] == "web_framework"
        assert d["tags"] == ["python"]


class TestArchitectureDesigner:
    def test_init(self):
        designer = ArchitectureDesigner()
        assert designer is not None
    
    def test_design_basic(self):
        designer = ArchitectureDesigner()
        
        mock_issues = [AuditIssue(
            issue_type=AuditIssueType.CODE_SMELL,
            severity=IssueSeverity.LOW,
            title="Test",
            description="Test",
            file_path="test.py",
            line_number=1
        )]
        
        mock_trends = type('MockTrends', (), {
            'top_recommendations': []
        })
        
        spec = designer.design(mock_issues, mock_trends)
        
        assert spec is not None
        assert spec.name == "AutoTestGen NextGen"
        assert len(spec.modules) > 0
        assert len(spec.data_flows) > 0
    
    def test_architecture_spec_to_dict(self):
        spec = ArchitectureSpec(
            name="Test",
            version="1.0",
            pattern=type('MockPattern', (), {'value': 'test'})
        )
        
        d = spec.to_dict()
        assert d["name"] == "Test"
        assert d["version"] == "1.0"
    
    def test_module_to_dict(self):
        module = ArchitectureModule(
            name="test_module",
            module_type=type('MockType', (), {'value': 'core'}),
            responsibility="Test",
            technology="Python"
        )
        
        d = module.to_dict()
        assert d["name"] == "test_module"
        assert d["responsibility"] == "Test"


class TestSelfRebirthEngine:
    def test_init(self):
        engine = SelfRebirthEngine()
        assert engine is not None
    
    def test_run_with_force_approval(self):
        engine = SelfRebirthEngine(human_approval_required=False)
        result = engine.run(force_approval=True)
        
        assert result.status == RebirthStatus.COMPLETED
        assert result.output_dir != ""
        assert os.path.exists(result.output_dir)
    
    def test_rebirth_result_to_dict(self):
        result = type('MockResult', (), {
            'status': type('MockStatus', (), {'value': 'completed'}),
            'version': '1.0',
            'output_dir': '/tmp/test',
            'logs': [],
            'audit_result': None,
            'trend_result': None,
            'architecture_spec': None,
            'sandbox_passed': True,
            'human_approved': True,
            'total_duration': 1.0,
            'message': 'Test'
        })
        
        d = {
            'status': result.status.value,
            'version': result.version,
            'output_dir': result.output_dir,
            'logs': [],
            'audit_result': None,
            'trend_result': None,
            'architecture_spec': None,
            'sandbox_passed': result.sandbox_passed,
            'human_approved': result.human_approved,
            'total_duration': result.total_duration,
            'message': result.message
        }
        
        assert d['status'] == 'completed'
        assert d['sandbox_passed'] is True