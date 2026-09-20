import os
import argparse
import glob
import sys
import json
import logging
from typing import Dict, List, Tuple

import pyapriltags
import cv2
import numpy as np


def calib_camera(
        trags_3d_list:List[pyapriltags.Tag3D],
        trag_3d_list_list:List[List[pyapriltags.Tag2D]],
        image_size:Tuple[int,int]        
)->Tuple[List[float],List[float]]:
    
