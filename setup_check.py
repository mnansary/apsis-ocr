from coreLib.ocr import OCR
from coreLib.utils import LOG_INFO
try:
    ocr=OCR()
    result,text=ocr("uploads/test.png")
except Exception as e:
    LOG_INFO("setup failed",mcolor="red")
    LOG_INFO("--------------------------------------------------------------------------------------------",mcolor="red")
    LOG_INFO(e,mcolor="green")
    LOG_INFO("--------------------------------------------------------------------------------------------",mcolor="red")
    