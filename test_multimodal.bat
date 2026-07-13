@echo off
echo ==============================================
echo 测试联邦多模态学习模块
echo ==============================================
echo.

cd /d "d:\PyCharmMiscProject\hermes-cosmos"

echo 1. 检查Python环境...
python --version
echo.

echo 2. 检查torch安装...
python -c "import torch; print('torch version:', torch.__version__)"
echo.

echo 3. 运行多模态学习测试...
python hermes_unified/run_multi_fl.py --direction multimodal
echo.

echo ==============================================
echo 测试完成！
echo ==============================================
pause