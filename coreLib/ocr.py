#-*- coding: utf-8 -*-
"""
@author:MD.Nazmuddoha Ansary
"""
from __future__ import print_function
from distutils.log import debug

#-------------------------
# imports
#-------------------------
from .utils import localize_box,LOG_INFO,download
from .detector import Detector
from .bnocr import BanglaOCR
from paddleocr import PaddleOCR
import os
import cv2
import copy
import pandas as pd

#-------------------------
# class
#------------------------

    
class OCR(object):
    def __init__(self,   
                 bnocr_onnx="weights/bnocr.onnx",
                 bnocr_gid="1YwpcDJmeO5mXlPDj1K0hkUobpwGaq3YA"):
        if not os.path.exists(bnocr_onnx):
            download(bnocr_gid,bnocr_onnx)
        self.bnocr=BanglaOCR(bnocr_onnx)
        LOG_INFO("Loaded Bangla Model")        
        self.line_en=PaddleOCR(use_angle_cls=True, lang='en',rec_algorithm='SVTR_LCNet')
        self.word_ar=PaddleOCR(use_angle_cls=True, lang='ar')
        self.det=Detector()
        LOG_INFO("Loaded Paddle detector")
        
        
    def process_boxes(self,word_boxes,line_boxes):

        # line_boxes
        line_orgs=[]
        line_refs=[]
        for bno in range(len(line_boxes)):
            tmp_box = copy.deepcopy(line_boxes[bno])
            x2,x1=int(max(tmp_box[:,0])),int(min(tmp_box[:,0]))
            y2,y1=int(max(tmp_box[:,1])),int(min(tmp_box[:,1]))
            line_orgs.append([x1,y1,x2,y2])
            line_refs.append([x1,y1,x2,y2])
        
        # merge
        for lidx,box in enumerate(line_refs):
            if box is not None:
                for nidx in range(lidx+1,len(line_refs)):
                    x1,y1,x2,y2=box    
                    x1n,y1n,x2n,y2n=line_orgs[nidx]
                    dist=min([abs(y2-y1),abs(y2n-y1n)])
                    if abs(y1-y1n)<dist and abs(y2-y2n)<dist:
                        x1,x2,y1,y2=min([x1,x1n]),max([x2,x2n]),min([y1,y1n]),max([y2,y2n])
                        box=[x1,y1,x2,y2]
                        line_refs[lidx]=None
                        line_refs[nidx]=box
                        
        line_refs=[lr for lr in line_refs if lr is not None]
        # sort line refs based on Y-axis
        line_refs=sorted(line_refs,key=lambda x:x[1])     
        # word_boxes
        word_refs=[]
        for bno in range(len(word_boxes)):
            tmp_box = copy.deepcopy(word_boxes[bno])
            x2,x1=int(max(tmp_box[:,0])),int(min(tmp_box[:,0]))
            y2,y1=int(max(tmp_box[:,1])),int(min(tmp_box[:,1]))
            word_refs.append([x1,y1,x2,y2])
            
        
        data=pd.DataFrame({"words":word_refs,"word_ids":[i for i in range(len(word_refs))]})
        # detect line-word
        data["lines"]=data.words.apply(lambda x:localize_box(x,line_refs))
        data["lines"]=data.lines.apply(lambda x:int(x))
        # register as crop
        text_dict=[]
        for line in data.lines.unique():
            ldf=data.loc[data.lines==line]
            _boxes=ldf.words.tolist()
            _bids=ldf.word_ids.tolist()
            _,bids=zip(*sorted(zip(_boxes,_bids),key=lambda x: x[0][0]))
            for idx,bid in enumerate(bids):
                _dict={"line_no":line,"word_no":idx,"crop_id":bid,"poly":word_boxes[bid]}
                text_dict.append(_dict)
        data=pd.DataFrame(text_dict)
        return data
    
    def __call__(self,img_path):
        result=[]
        # -----------------------start-----------------------
        img=cv2.imread(img_path)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        # text detection
        line_boxes,_=self.det.detect(img,self.line_en)
        word_boxes,crops=self.det.detect(img,self.word_ar)
        # line-word sorting
        df=self.process_boxes(word_boxes,line_boxes)
        # language classification
        cids=df.crop_id.tolist()
        word_crops=[crops[i] for i  in cids]
        #--------------------------------bangla------------------------------------
        bn_text=self.bnocr(word_crops)
        df["text"]=bn_text
        df=df.sort_values('line_no')
        # format
        for idx in range(len(df)):
            data={}
            data["line_no"]=int(df.iloc[idx,0])
            data["word_no"]=int(df.iloc[idx,1])
            # array 
            poly_res=  []
            poly    =  df.iloc[idx,3]
            for pair in poly:
                _pair=[float(pair[0]),float(pair[1])]
                poly_res.append(_pair)
            
            data["poly"]   =poly_res
            data["text"]   =df.iloc[idx,4]
            result.append(data)
        # lines
        df=pd.DataFrame(result)
        df=df[["text","line_no","word_no"]]
        lines=[]
        for line in df.line_no.unique():
            ldf=df.loc[df.line_no==line]
            ldf.reset_index(drop=True,inplace=True)
            ldf=ldf.sort_values('word_no')
            _ltext=''
            for idx in range(len(ldf)):
                text=ldf.iloc[idx,0]
                _ltext+=' '+text
            lines.append(_ltext)
        text="\n".join(lines)
        return result,text