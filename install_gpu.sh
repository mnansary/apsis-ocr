
#!/bin/sh
conda install cudatoolkit
pip install flask flask_cors termcolor gdown numpy bnunicodenormalizer streamlit
pip install onnxruntime-gpu
pip uninstall protobuf -y
pip install --no-binary protobuf protobuf==3.20.0
pip install paddlepaddle-gpu -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install "paddleocr>=2.0.1"
python setup_check.py

