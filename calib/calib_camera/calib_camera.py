import os
import argparse
import glob
import sys
import json
import logging
from typing import Dict, List, Tuple
import logging

import pyapriltags
import cv2
import numpy as np

logger = logging.getLogger(__name__)

def calib_camera(
        trags_3d_list:List[pyapriltags.Tag3D],
        trag_2d_list_list:List[List[pyapriltags.Tag2D]],
        image_size:Tuple[int,int]        
)->Tuple[List[float],List[float]]:
    
    pts3d_list=[]
    pts2d_list=[]

    for tag2d_list in trag_2d_list_list:

        pts2d=[]
        pts3d=[]

        for tag2d in tag2d_list:
            tag2d_id=tag2d.id
            tag3d=next(t for t in trags_3d_list if t.id==tag2d_id)
            if tag3d is None:
                logger.warning(
                    f"tagid={tag2d_id} only detected 2d_id skipping it"
                )
                continue

            for i in range(4):
                pts3d.append(np.array(tag3d))
                pts2d.append(np.array(tag2d))
        if len(pts3d)<4:
            logger.warning("valid 3D-2D point pairs found")
            continue

        pts3d_list.append(np.array(pts3d))
        pts2d_list.append(np.array(pts2d))

        if len(pts3d_list)<10:
            logger.warning("valid images with sufficient 3D-2D point pairs found")

            return None,None
        


        

    
