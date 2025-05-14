#!/usr/bin/python3
"""
author: Ross
date: Sept 18, 2020
Description:
Place for all constants
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm



# =========== Run Flags ===========
# Process_WebRTC = False # Flag to process the Webrtc logs
# Process_QoE= False # Flag to process per-frame-network delay
Process_Xcal = True # Flag to process the xcal data
# process_HO_Analysis = True

# HO Analysis Window
win = 10 # In seconds

processConfig = True

Total_Runs = 1

columns = ['TIME_STAMP', 'measurementReport eventType', 'Serving PCI',
       'Event Technology', 'Event Technology(Band)', 'CQI',
       'Event 5G-NR Events', 'Event 5G-NR/LTE Events',
       'UL MCS (Mode)', 'UL RB Num (Including 0)', 'RSRP [dBm]',
       'RSRQ [dB]', 'PUSCH Throughput [Mbps]']
# ul_cols = ['targetBitrate_Report-Macro_5', 'framesPerSecond_Report-Macro_5',
#            'framesPerSecond_Report-Macro_7', 'Mbps', 'ts']
ul_cols = ['ts', 'Mbps', 'framesPerSecond']

# Colors Definations
# get colors
cmap10 = plt.cm.tab10  # define the colormap
# extract all colors from the .jet map
colorlist10 = [cmap10(i) for i in range(cmap10.N)]

# get colors
cmap20 = plt.cm.tab20c  # define the colormap
# extract all colors from the .jet map
colorlist20 = [cmap20(i) for i in range(cmap20.N)]

# get colors
pastel2 = plt.cm.Pastel2  # define the colormap
# extract all colors from the .jet map
pastel2list = [pastel2(i) for i in range(pastel2.N)]

# get colors
paired = plt.cm.Paired  # define the colormap
# extract all colors from the .jet map
pairedlist = [paired(i) for i in range(paired.N)]

inferno = cm.get_cmap('inferno', 5)
infernolist = [inferno(i) for i in range(inferno.N)]

# colors = {'5G-NSA': 'lightgray', 'Lte': colorlist20[2], 'Orange-Spain': colorlist20[3], 'CQI': 'lightblue',
#           'roam': (0.8, 0.4, 0.5, 0.8), 'nonRoam': (0.5, 0.5, 0.2, 0.9), 'E2E_RTT': (0.1, 0.4, 0.7, 0.8),
#           'sp12': colorlist20[7], 'VodafUL': '#3a0856',
#           'DL': '#a39c76', 'UL': '#c9b030',
#           '5G-NSA-mmWave' : colorlist10[3],
#           'sp11': colorlist20[7], 'TMobileDL': '#412c11',
#           'CQI-roam': colorlist20[7], 'CQI-nonroam': colorlist20[2],
#           'Google': '#1c747a', 'Amazon': '#7a221c',
#           'SFR-1': '#db7461', 'MSF Azure': '#61c8db',
#           'it1': '#61db74', 'fr2': '#7461db',
#           'ger2': '#fbb7b4', 'walk': '#f5f4ed', 'stat': '#addcd6',
#           'Germany': '#E95C20FF', 'sp2':'#006747FF', 'France': '#4F2C1DFF'}

configs = ['SP11', 'SP12', 'SP2', 'FR1', 'FR2', 'IT1', 'IT2', 'IT3', 'GER1', 'GER2',
           'US1', 'US2', 'US3', 'US1_NoCA', 'US2_NoCA', 'US3_NoCA']

colors = {
    configs[0]: '#469990',
    configs[1]: '#808000',
    configs[2]: '#000075',
    configs[3]: '#dcbeff',
    configs[4]: '#800000',
    configs[5]: '#f58231',
    configs[6]: '#f58231',
    configs[7]: '#000000',
    configs[8]: '#f032e6',
    configs[9]: '#911eb4',
    configs[10]: '#3cb44b',
    configs[11]: '#9A6324',
    configs[12]: '#42d4f4',
    configs[13]: '#3cb44b',
    configs[14]: '#9A6324',
    configs[15]: '#42d4f4',
    'noShare': '#800000'
}

COLORS = {
    configs[0]: '#469990',
    configs[1]: '#808000',
    configs[2]: '#000075',
    configs[3]: '#dcbeff',
    configs[4]: '#800000',
    configs[5]: '#f58231',
    configs[6]: '#f58231',
    configs[7]: '#000000',
    configs[8]: '#f032e6',
    configs[9]: '#911eb4',
    configs[10]: '#3cb44b',
    configs[11]: '#9A6324',
    configs[12]: '#42d4f4',
    configs[13]: '#3cb44b',
    configs[14]: '#9A6324',
    configs[15]: '#42d4f4',
    'noShare': '#800000'
}

pfn_delay = '#FF00FF'
br = '#408ABE'

#EFCC00
# MARKERS = {
#     'sp11': '*',
#     'sp12': 'd',
#     'sp2': '>',
#     'VodafLTE': '>',
#     'VodafNR': '*',
#     'OrangeNR': 'v'
# }
MARKERS = {
    configs[0]: 'X',
    configs[1]: '^',
    configs[2]: '>',
    configs[3]: '3',
    configs[4]: '1',
    configs[5]: 'd',
    configs[6]: '*',
    configs[7]: 'p',
    configs[8]: '<',
    configs[9]: 'o',
    configs[10]: 'v',
    configs[11]: '.',
    configs[12]: 'P'
}

HATCHES = {
    configs[0]: '/o',
    configs[1]: '\\|',
    configs[2]: '|*',
    configs[3]: '-\\',
    configs[4]: 'O.',
    configs[5]: 'x*',
    configs[6]: 'o-',
    configs[7]: 'O|',
    configs[8]: '+o',
    configs[9]: '*-',
    configs[10]: '///',
    configs[11]: 'o',
    configs[12]: 'X',
}


LS = ['--', ':', '-.', '-']

cols_to_plot = ['TIME_STAMP', 'Per-Frame-Network-Delay-Seconds', 'Event 5G-NR Events', 'per-frame-delay',
               'RSRP [dBm]', 'RSRQ [dB]', 'Serving PCI', 'UL MCS (Mode)', 'CQI', 'UL RB Num (Including 0)',
                'PUSCH Throughput [Mbps]']


color_map = {'eventA3': 'purple',
             'PRACH: Msg5': 'blue',
             'FRAGMENT_LOADING_COMPLETED': 'purple'}

