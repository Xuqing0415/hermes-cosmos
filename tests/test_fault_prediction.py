"""
Tests for fault prediction module
"""

import pytest
from unittest.mock import Mock, patch
import numpy as np

from hermes.fault_prediction.predictor import FaultPredictor, GPUMetrics


class TestFaultPredictor:
    """测试故障预测器"""

    @patch('onnxruntime.InferenceSession')
    def test_predict_output_range(self, mock_session):
        """测试预测输出概率范围在 [0, 1] 之间"""
        # 配置mock
        mock_session.return_value.get_inputs.return_value = [Mock(name='input')]
        mock_session.return_value.get_outputs.return_value = [Mock(name='output')]
        mock_session.return_value.run.return_value = [np.array([[[0.75]]], dtype=np.float32)]
        
        predictor = FaultPredictor(model_path="dummy.onnx")
        
        metrics = {
            'temperature': 75.0,
            'power': 180.0,
            'utilization': 90.0,
            'memory_usage': 85.0,
            'ecc_errors': 2,
            'fan_speed': 70.0
        }
        
        probability = predictor.predict(metrics)
        
        assert 0.0 <= probability <= 1.0, f"Probability should be in [0, 1], got {probability}"

    @patch('onnxruntime.InferenceSession')
    def test_predict_boundary_values(self, mock_session):
        """测试边界值输入时的输出"""
        mock_session.return_value.get_inputs.return_value = [Mock(name='input')]
        mock_session.return_value.get_outputs.return_value = [Mock(name='output')]
        mock_session.return_value.run.return_value = [np.array([[[1.5]]], dtype=np.float32)]
        
        predictor = FaultPredictor(model_path="dummy.onnx")
        
        metrics = {
            'temperature': 100.0,
            'power': 300.0,
            'utilization': 100.0,
            'memory_usage': 100.0,
            'ecc_errors': 100,
            'fan_speed': 100.0
        }
        
        probability = predictor.predict(metrics)
        
        assert probability == 1.0, "Probability should be clamped to 1.0"

    @patch('onnxruntime.InferenceSession')
    def test_predict_negative_boundary(self, mock_session):
        """测试负边界值输入时的输出"""
        mock_session.return_value.get_inputs.return_value = [Mock(name='input')]
        mock_session.return_value.get_outputs.return_value = [Mock(name='output')]
        mock_session.return_value.run.return_value = [np.array([[[-0.5]]], dtype=np.float32)]
        
        predictor = FaultPredictor(model_path="dummy.onnx")
        
        metrics = {
            'temperature': 0.0,
            'power': 0.0,
            'utilization': 0.0,
            'memory_usage': 0.0,
            'ecc_errors': 0,
            'fan_speed': 0.0
        }
        
        probability = predictor.predict(metrics)
        
        assert probability == 0.0, "Probability should be clamped to 0.0"

    @patch('onnxruntime.InferenceSession')
    def test_get_risk_level(self, mock_session):
        """测试风险等级划分"""
        mock_session.return_value.get_inputs.return_value = [Mock(name='input')]
        mock_session.return_value.get_outputs.return_value = [Mock(name='output')]
        
        predictor = FaultPredictor(model_path="dummy.onnx")
        
        assert predictor.get_risk_level(0.9) == "critical"
        assert predictor.get_risk_level(0.7) == "high"
        assert predictor.get_risk_level(0.5) == "medium"
        assert predictor.get_risk_level(0.3) == "low"
        assert predictor.get_risk_level(0.8) == "critical"
        assert predictor.get_risk_level(0.6) == "high"
        assert predictor.get_risk_level(0.4) == "medium"

    @patch('onnxruntime.InferenceSession')
    def test_normalize(self, mock_session):
        """测试归一化函数"""
        mock_session.return_value.get_inputs.return_value = [Mock(name='input')]
        mock_session.return_value.get_outputs.return_value = [Mock(name='output')]
        
        predictor = FaultPredictor(model_path="dummy.onnx")
        
        # 测试均值处的值应该归一化为0
        input_data = np.array([70.0, 150.0, 80.0, 75.0, 0.5, 60.0], dtype=np.float32)
        normalized = predictor.normalize(input_data)
        
        assert np.allclose(normalized, np.zeros(6)), "Mean values should normalize to 0"

    @patch('onnxruntime.InferenceSession')
    def test_predict_consistency(self, mock_session):
        """测试相同输入的预测结果一致性"""
        mock_session.return_value.get_inputs.return_value = [Mock(name='input')]
        mock_session.return_value.get_outputs.return_value = [Mock(name='output')]
        mock_session.return_value.run.return_value = [np.array([[[0.65]]], dtype=np.float32)]
        
        predictor = FaultPredictor(model_path="dummy.onnx")
        
        metrics = {
            'temperature': 75.0,
            'power': 160.0,
            'utilization': 85.0,
            'memory_usage': 80.0,
            'ecc_errors': 1,
            'fan_speed': 65.0
        }
        
        result1 = predictor.predict(metrics)
        result2 = predictor.predict(metrics)
        result3 = predictor.predict(metrics)
        
        assert result1 == result2 == result3, "Same input should produce same output"
