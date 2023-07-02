# apsis-ocr
APSIS OCR Repo

# **GPU-Inference**

**Environment Setup**

* **Installing conda**: 
    *  ```curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o Miniconda3-latest-Linux-x86_64.sh```
    *  ```bash Miniconda3-latest-Linux-x86_64.sh```

* **create a conda environment**: ```conda create -n apsisocr python=3.8.13```
* **activate conda environment**: ```conda activate apsisocr```
* **install gpu dependencies**  : ```./install_gpu.sh```  


**GPU INFERENCE SERVER CONFIG**  

```python
OS          : Ubuntu 20.04.4 LTS       
Memory      : 31.3 GiB 
Processor   : Intel® Core™ i9-10900K CPU @ 3.70GHz × 20    
Graphics    : NVIDIA GeForce RTX 3090/PCIe/SSE2
Gnome       : 3.36.8
```

# **Deployment**

```python
nohup python api_ocr.py & # deploys at port 3032 by defautl
nohup streamlit run app.py --server.port 3033 & # deploys streamlit built frontend at 3033 port
```