@echo off
echo Starting torch test...
py -c "import torch; print('torch version:', torch.__version__)"
echo Test completed!