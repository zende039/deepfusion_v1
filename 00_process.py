#!/usr/bin/python3
'''
author: Rushikesh
date: March 18, 2025
    This is a test.
    Description:
         This file processes the OokleSpeedTest files
'''

import numpy as np
import pandas as pd
import sys, os
import shutil
from tqdm import tqdm
tqdm.pandas()

sys.path.append(os.getcwd())
myPath = str(os.getcwd())
sys.path.append(os.path.abspath(os.path.join(myPath, os.pardir)))

# Add the Helpers directory
helpers_path = os.path.abspath(os.path.join(myPath, "Helpers"))
sys.path.append(helpers_path)

# Add the Script directory
scripts_path = os.path.abspath(os.path.join(myPath, "Scripts"))
sys.path.append(scripts_path)


from Helpers.helpers import *
from Helpers.utils import plotme
from Helpers.constants import *
from Helpers.set_paths import *

import matplotlib.pyplot as plt
import re
import seaborn as sns
import os
from datetime import datetime
from datetime import datetime
import matplotlib.gridspec as gridspec
from prettytable import PrettyTable

if Process_Xcal:
    flag = 'Xcal/'
    camera = 'with_RRC'
    dataFiles = getFiles()
    for f in tqdm(dataFiles):
        if camera:
            file_name = os.path.splitext(os.path.basename(f))[0]
            print('\t----> Processing Raw Data {} <----'.format(file_name))
            df = loadData(f)
            newDF = processXcal(df)
            print('\n\t\t----> Processing During <----')

            # All HO Process
            event = 'eventA3'
            processedDF_During = process_HO_During(newDF, event, 'PRACH: Msg5', 'HO Duration(secs)')
            # processedDF_During = process_HO_During(newDF, 'eventA3', r'RRC Setup Success///', 'HO Duration(secs)')




            saveFile(processedDF_During, PROCESSDATA_FOLDER, file_name + '_During_HO_' + event+'.txt')

            event = 'eventA5'
            processedDF_During = process_HO_During(newDF, event, 'PRACH: Msg5', 'HO Duration(secs)')
            # processedDF_During = process_HO_During(newDF, 'eventA3', r'RRC Setup Success///', 'HO Duration(secs)')




            saveFile(processedDF_During, PROCESSDATA_FOLDER, file_name + '_During_HO_' + event+'.txt')

            # print('\t\t----> Processing After HO <----')
            # processedDF_After = process_HO_Before_After(newDF, flag='after')
            # saveFile(processedDF_After, PROCESSDATA_FOLDER, file_name + '_After_HO.txt')
            # print('\t\t----> Processing Before HO <----')
            # processedDF_Before = process_HO_Before_After(newDF, flag='before', time_interval_seconds=2)
            # saveFile(processedDF_Before, PROCESSDATA_FOLDER, file_name + '_Before_HO.txt')
            print('\t----> Done Processing <----')

