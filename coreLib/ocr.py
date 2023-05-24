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
from .paddet import Detector
from .bnocr import BanglaOCR
from paddleocr import PaddleOCR
import os
import cv2
import numpy as np
import copy
import pandas as pd
import matplotlib.pyplot as plt
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
        
        self.base=PaddleOCR(use_angle_cls=True, lang='en',rec_algorithm='SVTR_LCNet',use_gpu=True)
        self.det=Detector()
        LOG_INFO("Loaded Paddle")

        
    def process_boxes(self,img,boxes):
        # boxes
        word_orgs=[]
        for bno in range(len(boxes)):
            tmp_box = copy.deepcopy(boxes[bno])
            x2,x1=int(max(tmp_box[:,0])),int(min(tmp_box[:,0]))
            y2,y1=int(max(tmp_box[:,1])),int(min(tmp_box[:,1]))
            word_orgs.append([x1,y1,x2,y2])
        
        # references
        line_refs=[]
        mask=create_mask(img,boxes)
        # Create rectangular structuring element and dilate
        mask=mask*255
        mask=mask.astype("uint8")
        h,w=mask.shape
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w//2,1))
        dilate = cv2.dilate(mask, kernel, iterations=4)

        # Find contours and draw rectangle
        cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            x,y,w,h = cv2.boundingRect(c)
            line_refs.append([x,y,x+w,y+h])
        line_refs = sorted(line_refs, key=lambda x: (x[1], x[0]))


        # organize       
        data=pd.DataFrame({"words":word_orgs,"word_ids":[i for i in range(len(word_orgs))]})
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
                _dict={"line_no":line,"word_no":idx,"crop_id":bid,"box":boxes[bid]}
                text_dict.append(_dict)
        df=pd.DataFrame(text_dict)
        return df
    #-------------------------------------------------------------------------------------------------------------------------
    # exectutives
    #-------------------------------------------------------------------------------------------------------------------------
    def get_coverage(self,image,mask):
        # -- coverage
        h,w,_=image.shape
        idx=np.where(mask>0)
        y1,y2,x1,x2 = np.min(idx[0]), np.max(idx[0]), np.min(idx[1]), np.max(idx[1])
        ht=y2-y1
        wt=x2-x1
        coverage=round(((ht*wt)/(h*w))*100,2)
        return coverage  


    def execute_rotation_fix(self,image,mask):
        image,mask,angle=auto_correct_image_orientation(image,mask)
        rot_info={"operation":"rotation-fix",
                  "optimized-angle":angle,
                  "text-area-coverage":self.get_coverage(image,mask)}

        return image,rot_info

            
    def __call__(self,
                 img_path,
                 face,
                 ret_bangla,
                 exec_rot,
                 coverage_thresh=30,
                 debug=False):
        # return containers
        data={}
        included={}
        executed=[]
        # -----------------------start-----------------------
        img=cv2.imread(img_path)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        src=np.copy(img)
    
        if face=="front":
            clss=['bname', 'ename', 'fname', 'mname', 'dob', 'nid']
        else:
            clss=['addr','back']
        
        try:
            # orientation
            if exec_rot:
                # mask
                mask,_,_=self.det.detect(img,self.base)
                img,rot_info=self.execute_rotation_fix(img,mask)
                executed.append(rot_info)
        except Exception as erot:
            print("--------------------------error----------------------------------------------------")
            print(erot)
            print("--------------------------error----------------------------------------------------")
            return "text-region-missing"
            
        # check yolo
        img,locs,founds=self.loc(img,clss,face)

        if img is None:
            if len(founds)==0:
                return "no-fields"
            else:
                if not exec_rot:
                    mask,_,_=self.det.detect(src,self.base)
                coverage=self.get_coverage(src,mask)
                if coverage > coverage_thresh:
                    return "loc-error"
                else:
                    return f"coverage-error#{coverage}"
        else:
            if face=="front":
                # text detection [word-based]
                _,boxes,crops=self.det.detect(img,self.base)
                # sorted box dictionary [clss based box_dict]
                box_dict=self.process_boxes(boxes,locs,clss)        
                data["nid-basic-info"]=get_basic_info(box_dict,crops,self.base)
                if ret_bangla:
                    included["bangla-info"]=get_bangla_info(box_dict,crops,self.bnocr)
            
            else:
                # crop image if both front and back is present
                if locs['dob'] is not None and locs["nid"] is not None:
                    img,locs=reformat_back_data(img,locs)
                # text detection 
                line_mask,line_boxes,_=self.det.detect(img,self.line)
                word_mask,word_boxes,word_crops=self.det.detect(img,self.base)
                # regional text
                line_boxes,word_boxes,crops=get_regional_box_crops(line_mask,line_boxes,word_mask,word_boxes,word_crops)
                texts=self.bnocr(crops)
                # get address
                data["nid-back-info"]=get_addr(word_boxes,line_boxes,texts)
                
                
            # containers
            data["included"]=included
            data["executed"]=executed
            return data 

