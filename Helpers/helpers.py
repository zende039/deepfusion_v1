#!/usr/bin/python3
'''
Author: Ross Fezeu
Date: Oct 18th, 2021.
    Description:
        Function module helpers for the processing
'''
import os.path
from tqdm import tqdm
tqdm.pandas()

# import glob
import pandas as pd
from Helpers.constants import *
# from datetime import datetime
from collections import defaultdict
# import pytz
# import os
import sys
import re
# import logging
from set_paths import *
# from constants import *
import pandas.io.common
from matplotlib.patches import Ellipse
from datetime import timedelta


def getFiles(flag='Raw'):
    # Ctry_Folder = getMobFolder(country)
    """ load the files """
    files_to_process = []
    folders_to_process = []
    # r=root, d=directories, f = files
    if flag == 'Raw':
        for r, d, f in os.walk(RAWDATA_FOLDER):
            for file in f:
                if file[0] != '.':
                    files_to_process.append(r + file)
    elif flag == 'Process':
        for r, d, f in os.walk(PROCESSDATA_FOLDER):
            for file in f:
                if file[0] != '.':
                    files_to_process.append(r + file)
    # elif flag == 'Vod':
    #     for r, d, f in os.walk(VOD_QoE_FOLDER):
    #         for file in f:
    #             if file[0] != '.':
    #                 files_to_process.append(r + '/' + file)
    # elif flag == 'TeleOp':
    #     for r, d, f in os.walk(TELE_OP):
    #         for file in f:
    #             if file[0] != '.':
    #                 files_to_process.append(r + '/' + file)
    # elif flag == 'Results':
    #     for r, d, f in os.walk(DATA_RESULTS):
    #         for file in f:
    #             if file[0] != '.':
    #                 files_to_process.append(r + '/' + file)
    # else:
    #     for r, d, f in os.walk(country):
    #         for file in f:
    #             if file[0] != '.':
    #                 files_to_process.append(r + '/' + file)
    if len(files_to_process) == 0:
        print("No New Files to Process. Script Ended")
        # logging.info('No New Files to Process. Script Ended')
        sys.exit(1)
    return files_to_process


def saveFile(df, folder, file):
    # if not os.path.exists(folder):
    if not os.path.isdir(folder):
        os.makedirs(folder)
    # print(folder + file)
    df.to_csv(folder + file, sep='\t', index=False)
    return


def loadTextFile(file):
    # mob_Folder = getMobFolder(mobility)
    file = open(file, 'r')
    return file.readlines()


def loadData(file, flag='none'):
    # if not os.path.exists(folder + "/" + file):
    if not os.path.exists(file):
        print('{} File does not exist in the {} directory'.format(file))
        exit(1)
    # if flag == 'detail':
    #     data = pd.read_csv('{}/{}'.format(file), names=Detail_File_Cols, sep=',', low_memory=False)
    # elif flag == 'frameInfo':
    #     data = pd.read_csv('{}'.format(file), names=Frame_Info_File, sep=',', low_memory=False)
    # if flag == 'none':
    data = pd.read_csv('{}'.format(file), sep='\t', low_memory=False, on_bad_lines='skip')
    return data


def processXcal(data):
    for col in tqdm(data.columns.tolist()):
        if (col == '5G KPI PCell RF Band' or
                col == '5G KPI PCell RF BandWidth' or col == '5G KPI PCell RF Frequency [MHz]' or
                col == '5G KPI PCell RF Subcarrier Spacing' or col == '5G KPI PCell RF Duplex Mode' or
        col == 'API GPS Info Altitude' or col == 'API GPS Info Speed' or col == '5G KPI PCell RF Serving PCI' or
        col == '5G KPI PCell RF Neighbor Top1 PCI' or col == '5G KPI SCell[1] RF Serving PCI' or
                col == '5G KPI SCell[1] RF Neighbor Top1 PCI'):
            data[col] = data[col].ffill()
    for col in tqdm(data.columns.tolist()):
        if '5G KPI PCell RF ' in col:
            data.columns = data.columns.str.replace(
                '5G KPI PCell RF ', 'PCell ', regex=True)
        if '5G KPI SCell[1] RF ' in col:
            data.columns = data.columns.str.replace(r'5G KPI SCell\[1\] RF ', 'SCell ', regex=True)
        if '5G-NR RRC NR MCG Mobility Statistics Intra-NR Handover' in col:
            data.columns = data.columns.str.replace(r'5G-NR RRC NR MCG Mobility Statistics Intra-NR Handover', 'Intra-NR HO ')
        if '5G-NR RRC NR MCG Mobility Statistics Intrafreq Handover ' in col:
            data.columns = data.columns.str.replace(r'5G-NR RRC NR MCG Mobility Statistics Intrafreq Handover ', 'HO ')
        if 'Event Technology' in col:
            data.columns = data.columns.str.replace(r'Event Technology', 'Tech',regex=True)
            data['Tech'] = data['Tech'].ffill()
            data['Tech(Band)'] = data['Tech(Band)'].ffill()
        if 'Lat' in data.columns:
            data = data.drop(['Lat', 'Lon'], axis=1)
        # if col == 'Event Technology(Band)':
        #     data.columns = data.columns.str.replace(r'Event Technology(Band)', 'Tech Band', regex=True)
        #     data['Tech Band'] = data['Tech Band'].ffill()
    data['TIME_STAMP'] = pd.to_datetime(data['TIME_STAMP'], errors='coerce')
    data['TIME_STAMP'] = data['TIME_STAMP'].dt.tz_localize('UTC')
    data['TIME_STAMP'] = data['TIME_STAMP'].dt.tz_convert('Asia/Karachi')
    data['TIME_STAMP'] = data['TIME_STAMP'].dt.tz_localize(None)
    data['ts'] = data['TIME_STAMP'].dt.strftime('%Y-%m-%d %H:%M:%S.%f')  # [:]

        # if 'tdd-UL-DL-ConfigurationCommon' in col:
        #     data.columns = data.columns.str.replace(
        #         r'^secondaryCellGroup spCellConfig reconfigurationWithSync spCellConfigCommon tdd-UL-DL-ConfigurationCommon ', '',
        #         regex=True)
        # if 'Qualcomm 5G-NR PDCP Throughput NR-' in col:
        #     data.columns = data.columns.str.replace('Qualcomm 5G-NR PDCP Throughput NR-', '', regex=True)
        # if 'Qualcomm 5G-NR ML1 Searcher Measurement PCell Serving Cell Serving Cell Level ' in col:
        #     data.columns = data.columns.str.replace('Qualcomm 5G-NR ML1 Searcher Measurement PCell Serving Cell Serving Cell Level ', '', regex=True)
        # if col == '5G KPI PCell RF Serving PCI':
        #     data.columns = data.columns.str.replace('5G KPI PCell RF ', '', regex=True)
        #     data['Serving PCI'] = data['Serving PCI'].ffill()
        # if '5G KPI PCell Layer1 ' in col:
        #     data.columns = data.columns.str.replace('5G KPI PCell Layer1 ', '', regex=True)
        # if '5G KPI PCell RF ' in col:
        #     data.columns = data.columns.str.replace('5G KPI PCell RF ', '', regex=True)
        #     data['CQI'] = data['CQI'].ffill()
        # if '5G KPI Total Info Layer1 ' in col:
        #     data.columns = data.columns.str.replace('5G KPI Total Info Layer1 ', '', regex=True)
        # if col == 'Event Technology':
        #     data['Event Technology'] = data['Event Technology'].ffill()
        # if col == 'Event Technology(Band)':
        #     data['Event Technology(Band)'] = data['Event Technology(Band)'].ffill()


    # data.columns = data.columns.str.replace(
    #     'Smart Phone Android System Info Operator', 'Operator', regex=True)
    # data.columns = data.columns.str.replace(
    #     'Smart Phone Android Mobile Info Service Provider Name', 'Operator', regex=True)
    # if '5G KPI PCell RF CQI' in data.columns:
    #     data.columns = data.columns.str.replace(
    #         '5G KPI PCell RF CQI', 'CQI', regex=True)
    #     data['CQI'] = data['CQI'].ffill()
    # if 'Ping & Trace Request' in data.columns:
    #     data.columns = data.columns.str.replace(
    #         'Ping & Trace Request', 'Ping Request', regex=True)
    #     data['Ping Request'] = data['Ping Request'].ffill()
    # if 'Ping & Trace Response' in data.columns:
    #     data.columns = data.columns.str.replace(
    #         'Ping & Trace Response', 'Ping Response', regex=True)
    #     data['Ping Response'] = data['Ping Response'].ffill()
    # if '5G-NR RRC NR SA+NSA State Info RRC State' in data.columns:
    #     data['RRC State'] = data['5G-NR RRC NR SA+NSA State Info RRC State']
    #     data['RRC State'] = data['RRC State'].ffill()
    #     data = data.drop(['5G-NR RRC NR SA+NSA State Info RRC State'], axis=1)
    # data.columns = data.columns.str.replace(
    #     r'^Event Autocall Scenario Name', 'Autocall',
    #     regex=True)
    # data.columns = data.columns.str.replace(
    #     r'^secondaryCellGroup spCellConfig reconfigurationWithSync spCellConfigCommon downlinkConfigCommon frequencyInfoDL ', '', regex=True)
    # data.columns = data.columns.str.replace(
    #     r'^scs\-SpecificCarrierList List\[0\] ', '', regex=True)
    # # if 'Qualcomm 5G-NR MAC CDRX Config Info On_Duration_Time [ms]' in data.columns:
    # #     data['CDRX On Duration'] = data['Qualcomm 5G-NR MAC CDRX Config Info On_Duration_Time [ms]']
    # #     data['CDRX On Duration'] = data['CDRX On Duration'].ffill()
    # #     data = data.drop(['Qualcomm 5G-NR MAC CDRX Config Info On_Duration_Time [ms]'], axis=1)
    # # if 'Qualcomm 5G-NR MAC CDRX Config Info Inactivity_Timer [ms]' in data.columns:
    # #     data['CDRX Inactivity Duration'] = data['Qualcomm 5G-NR MAC CDRX Config Info Inactivity_Timer [ms]']
    # #     data['CDRX Inactivity Duration'] = data['CDRX Inactivity Duration'].ffill()
    # #     data = data.drop(['Qualcomm 5G-NR MAC CDRX Config Info Inactivity_Timer [ms]'], axis=1)
    # if 'LTE KPI PCell WB CQI CW0' in data.columns:
    #     data.columns = data.columns.str.replace(
    #         'LTE KPI PCell WB CQI CW0', 'CQI', regex=True)
    #     data['CQI'] = data['CQI'].ffill()


    # for col in data.columns.tolist():
    #     if 'Qualcomm 5G-NR MAC PDSCH' in col:
    #         data.columns = data.columns.str.replace(
    #             r'^Qualcomm 5G-NR MAC PDSCH Status\[Per Slot\] PCell ', '', regex=True)
    #         break
    #     if 'Qualcomm 5G-NR MAC PUSCH' in col:
    #         data.columns = data.columns.str.replace(
    #             r'^Qualcomm 5G-NR MAC PUSCH Info PCell ', '', regex=True)
    #         break
    #     elif 'Qualcomm 5G-NR MAC DCI' in col:
    #         if 'Qualcomm 5G-NR MAC DCI Info[Per Slot] PCell' in col:
    #             data.columns = data.columns.str.replace(
    #                 r'^Qualcomm 5G-NR MAC DCI Info\[Per Slot\] PCell ','', regex=True)
    #             break
    #         elif 'Qualcomm 5G-NR MAC DCI Info\_Ver2\[Per Slot\] PCell':
    #             data.columns = data.columns.str.replace(
    #                 r'^Qualcomm 5G-NR MAC DCI Info\_Ver2\[Per Slot\] PCell ','', regex=True)
    #             data.columns = data.columns.str.replace(
    #                 r'^Qualcomm 5G-NR MAC DL DCI Info\_Ver2\[Per Slot\] PCell ','', regex=True)
    #             for col in data.columns.tolist():
    #                 if 'Aggregation Level' in col:
    #                     data['Aggregation level'] = data['Aggregation Level']
    #                     break
    #             break
    #     elif 'Qualcomm 5G-NR MAC CDRX' in col:
    #         data.columns = data.columns.str.replace(
    #             r'^Qualcomm 5G-NR MAC CDRX Config ','', regex=True)
    #         break
    #     elif 'ML1 CA Metrics Config Info Carrier Configuration' in col:
    #         data.columns = data.columns.str.replace(
    #             r'^ML1 CA Metrics Config Info Carrier Configuration ','', regex=True)
    #         break
    #     elif '5G KPI PCell RF Subcarrier Spacing' in col:
    #         data.columns = data.columns.str.replace(
    #             r'^ML1 CA Metrics Config Info Carrier Configuration ','', regex=True)
    #         break

    # # Process columns and fill forward
    # if 'Event Technology' in data.columns:
    #     data['Event Technology'] = data['Event Technology'].ffill()
    # if 'Operator' in data.columns:
    #     data['Operator'] = data['Operator'].ffill()
    #     # data['SIM'] = data['Operator']
    # if 'pucch-Config setup dl-DataToUL-ACK' in data.columns or 'secondaryCellGroup spCellConfig spCellConfigDedicated uplinkConfig uplinkBWP-ToAddModList List[0] bwp-Dedicated pucch-Config setup dl-DataToUL-ACK' in data.columns\
    #         or 'secondaryCellGroup spCellConfig spCellConfigDedicated uplinkConfig initialUplinkBWP pucch-Config setup dl-DataToUL-ACK' in data.columns:
    #     data['5G KPI PCell RF Subcarrier Spacing'] = data['5G KPI PCell RF Subcarrier Spacing'].ffill()
    # if '5G KPI PCell RF BandWidth' in data.columns:
    #     data = data.drop(['Qualcomm Lte/LteAdv Configuration Info DL Common UL Config UL EARFCN', 'Qualcomm Lte/LteAdv Configuration Info DL Common UL Config UL Bandwidth'], axis=1)
    #     data['5G KPI PCell RF Band'] = data['5G KPI PCell RF Band'].ffill()
    #     data['5G KPI PCell RF Subcarrier Spacing'] = data['5G KPI PCell RF Subcarrier Spacing'].ffill()
    #     data['5G KPI PCell RF BandWidth'] = data['5G KPI PCell RF BandWidth'].ffill()
    #     data['5G KPI PCell RF Frequency [MHz]'] = data['5G KPI PCell RF Frequency [MHz]'].ffill()
    # if '5G KPI PCell RF Duplex Mode' in data.columns:
    #     data['5G KPI PCell RF Duplex Mode'] = data['5G KPI PCell RF Duplex Mode'].ffill()
    # if '5G KPI PCell Layer1 DL Layer Num (Mode)' in data.columns:
    #     data['5G KPI PCell Layer1 DL Layer Num (Mode)'] = data['5G KPI PCell Layer1 DL Layer Num (Mode)'].fillna(
    #         method='ffill')
    # if '5G KPI PCell Layer1 DL Modulation0 Representative Value' in data.columns:
    #     data['5G KPI PCell Layer1 DL Modulation0 Representative Value'] = data[
    #         '5G KPI PCell Layer1 DL Modulation0 Representative Value'].ffill()
    # if 'Autocall' in data.columns:
    #     data['Autocall'] = data['Autocall'].ffill()
    # if 'Info Inactivity_Timer [ms]' in data.columns:
    #     dataf['Info Inactivity_Timer [ms]'] = data['Info Inactivity_Timer [ms]'].ffill()
    # if 'Info On_Duration_Time [ms]' in data.columns:
    #     data['Info On_Duration_Time [ms]'] = data['Info On_Duration_Time [ms]'].ffill()
        # data['SIM'] = data['Operator']
    # data = data.dropna().reset_index(drop=True)

    # data = data.drop_duplicates()
    # data = data.dropna().reset_index(drop=True)
    return data


def get_subframe_by_time_interval(df, index, time_interval_seconds, direction='before'):
    """ Helper function to get a subframe based on a time interval before or after a specific row. """
    # Adjust the timestamp based on the direction (before or after)
    if direction == 'before':
        target_row = df.iloc[index + 1]
        timestamp = pd.to_datetime(target_row['TIME_STAMP'])
        time_window_start = timestamp - timedelta(seconds=time_interval_seconds)
        subframe = df[(df['TIME_STAMP'] >= time_window_start) & (df['TIME_STAMP'] <= timestamp)]
    elif direction == 'after':
        target_row = df.iloc[index - 1]
        timestamp = pd.to_datetime(target_row['TIME_STAMP'])
        time_window_end = timestamp + timedelta(seconds=time_interval_seconds)
        subframe = df[(df['TIME_STAMP'] >= timestamp) & (df['TIME_STAMP'] <= time_window_end)]

    return subframe


def process_HO_Before_After(df, flag='after', time_interval_seconds=1):
    """ Process the subframe 5 seconds after the 'Success' row and apply similar transformations. """
    columns_to_skip = ['TIME_STAMP', 'ts']

    # 1. Get the subframe based on values in the "Event 5G-NR Events" column for "Success"
    success_df = df[df['Intra-NR HO Duration [sec]'].notna()]
    # success_df = df[df['Event 5G-NR Events'] == 'NR Intrafreq Handover Success']

    # Initialize an empty list to hold updated DataFrames
    updated_dfs = []

    # Process each "Success" row
    for _, success_row in success_df.iterrows():
        success_index = success_row.name

        # 2. Get the subframe for 5 seconds after the 'Success' row
        if flag == 'after':
            prefix = "_a"
            subframe_a = get_subframe_by_time_interval(df, success_index, time_interval_seconds, direction='after')
        elif flag == 'before':
            prefix = "_b"
            subframe_a = get_subframe_by_time_interval(df, success_index, time_interval_seconds, direction='before')

        # Fill values based on rules for all other columns (except TIME_STAMP, ts, and Intra-NR HO Duration [Sec])
        for col in df.columns:
            if col != 'Intra-NR HO Duration [sec]' and col not in columns_to_skip:
                # Check if the column is numeric (int or float)
                if pd.api.types.is_numeric_dtype(subframe_a[col]):
                    avg_value = subframe_a[col].mean(skipna=True)
                    # Replace only where Intra-NR HO Duration [Sec] is non-NaN
                    subframe_a.loc[subframe_a['Intra-NR HO Duration [sec]'].notna(), col] = avg_value
                else:
                    # For categorical columns, use the most frequent (mode) value, but check if there is a mode value
                    mode_values = subframe_a[col].mode(dropna=True)
                    if not mode_values.empty:  # Check if there are any mode values
                        unique_value = mode_values[0]  # Most frequent value (mode)
                        # Replace only where Intra-NR HO Duration [Sec] is non-NaN
                        subframe_a.loc[subframe_a['Intra-NR HO Duration [sec]'].notna(), col] = unique_value
                    else:
                        # If there's no mode (i.e., all values are NaN or empty), leave the column unchanged
                        subframe_a.loc[subframe_a['Intra-NR HO Duration [sec]'].notna(), col] = subframe_a[col]

        # Append the updated subframe to the list
        updated_dfs.append(subframe_a[(subframe_a['Intra-NR HO Duration [sec]'].notna())])
        # updated_dfs.append(subframe_a)

    # Combine all updated DataFrames
    updated_df = pd.concat(updated_dfs)

    # 3. Return only rows where 'Intra-NR HO Duration [Sec]' is non-zero and not NaN
    updated_df = updated_df[(updated_df['Intra-NR HO Duration [sec]'].notna()) & (updated_df['Intra-NR HO Duration [sec]'] != 0)]

    # 4. Add '_a' suffix to all column names (except for 'Intra-NR HO Duration [sec]' and columns to skip)
    # updated_df.columns = [col + prefix if col != 'Intra-NR HO Duration [sec]' and col not in columns_to_skip else col
    #                       for col in updated_df.columns]

    return updated_df


def process_HO_Before(df, time_interval_seconds=5):
    """ Process the subframe 5 seconds before the 'Attempt' row and apply similar transformations. """
    columns_to_skip = ['TIME_STAMP', 'ts']

    # Initialize an empty list to hold updated DataFrames
    updated_dfs = []

    # Process each "Attempt" row
    for _, attempt_row in df[df['Tech'] == 'Attempt'].iterrows():
        attempt_index = attempt_row.name

        # Get subframe for 5 seconds before the 'Attempt' row
        subframe_b = get_subframe_by_time_interval(df, attempt_index, time_interval_seconds, direction='before')

        # Update column names with "_b" prefix
        subframe_b.columns = [col + '_b' if col not in columns_to_skip else col for col in subframe_b.columns]

        # Apply similar logic to fill missing values
        for col in subframe_b.columns:
            if col != 'Intra-NR HO Duration [Sec]_b' and col not in columns_to_skip:
                if pd.api.types.is_numeric_dtype(subframe_b[col]):
                    avg_value = subframe_b[col].mean(skipna=True)
                    subframe_b[col] = np.where(subframe_b['Intra-NR HO Duration [Sec]_b'].notna(), avg_value,
                                               subframe_b[col])
                else:
                    mode_values = subframe_b[col].mode(dropna=True)
                    if not mode_values.empty:
                        unique_value = mode_values[0]
                        subframe_b[col] = np.where(subframe_b['Intra-NR HO Duration [Sec]_b'].notna(), unique_value,
                                                   subframe_b[col])

        updated_dfs.append(subframe_b)

    return pd.concat(updated_dfs)


def process_HO_During(df, from_colName, to_colName, colName):
    # Columns to skip during processing
    columns_to_skip = ['TIME_STAMP', 'ts']

    # 1. Get the sub-dataframe based on values in the "Tech" column for "Attempt" and "Success"
    # attempt_df = df[df['Event 5G-NR Events'] == 'NR Intrafreq Handover Attempt']
    # success_df = df[df['Event 5G-NR Events'] == 'NR Intrafreq Handover Success']


    attempt_df = df[df['Event 5G-NR Events'] == from_colName]
    success_df = df[df['Event 5G-NR Events'] == to_colName]

    # rrc_col = '5G-NR RRC CallConnectionControl RRCConnectionProcedure rrcSetupComplete'
    # success_df = df[df[rrc_col] == to_colName]

    df[colName] = 0

    # Initialize an empty list to hold updated DataFrames
    updated_dfs = []

    # Process each group between "Attempt" and "Success"
    for _, attempt_row in attempt_df.iterrows():
        # Find corresponding success row (following attempt row)
        attempt_index = attempt_row.name
        success_index = success_df[success_df.index > attempt_index].iloc[0].name

        # Get the subframe between attempt and success
        subframe = df.loc[attempt_index:success_index]

        if 'NR Intrafreq Handover Attempt' not in subframe['Event 5G-NR Events'].unique().tolist() or 'NR Intrafreq Handover Success' not in subframe['Event 5G-NR Events'].unique().tolist():
            continue

        # Fill values based on rules for all other columns (except TIME_STAMP, ts, and Intra-NR HO Duration [Sec])
        for col in df.columns:
            # Skip the target column itself and the specified columns to skip
            if col != colName and col not in columns_to_skip and col != 'PCell Serving PCI' and col != 'Event 5G-NR Events':
                # Check if the column is numeric (int or float)
                if pd.api.types.is_numeric_dtype(subframe[col]):
                    # If the column is numeric, calculate the mean, excluding NaN values
                    avg_value = subframe[col].mean(skipna=True)
                    # Replace only where Intra-NR HO Duration [Sec] is non-NaN
                    # df.loc[subframe.index, col] = subframe['Intra-NR HO Duration [sec]'].notna() * avg_value  # Replace just the rows where Intra-NR HO Duration [Sec] is non-NaN
                    subframe.loc[success_index, col] = avg_value  # Replace just the rows where Intra-NR HO Duration [Sec] is non-NaN
                else:
                    # For categorical columns, use the most frequent (mode) value, but check if there is a mode value
                    mode_values = subframe[col].mode(dropna=True)
                    if not mode_values.empty:  # Check if there are any mode values
                        unique_value = mode_values[0]  # Most frequent value (mode)
                        # Replace only where Intra-NR HO Duration [Sec] is non-NaN
                        # df.loc[subframe.index, col] = np.where(subframe['Intra-NR HO Duration [sec]'].notna(),unique_value, subframe[col])
                        subframe.loc[success_index, col] = unique_value
                    else:
                        # If there's no mode (i.e., all values are NaN or empty), leave the column unchanged
                        # df.loc[success_index, col] = subframe[col]  # Leave the column as it is for rows with all NaN values
                        subframe.loc[success_index, col] = subframe.loc[success_index, col] # Leave the column as it is for rows with all NaN values

        # Only update 'Intra-NR HO Duration [Sec]' column if its value is non-NaN
        # df.loc[subframe.index, 'Intra-NR HO Duration [sec]'] = subframe['Intra-NR HO Duration [sec]'].apply(
        #     lambda x: avg_value if pd.notna(x) else x
        # )


        # Append the updated DataFrame to the list
        subframe.loc[success_index, colName] = (subframe['TIME_STAMP'][success_index] - subframe['TIME_STAMP'][attempt_index]).total_seconds()

        # Computing T1: eventA3 -- HO Attempt
        t1_df = subframe[subframe['Event 5G-NR Events'] == 'NR Intrafreq Handover Attempt']
        subframe.loc[success_index, 'T1_'+from_colName+'-'+'HO Attempt'] = (t1_df['TIME_STAMP'][t1_df.index[0]] - subframe['TIME_STAMP'][attempt_index] ).total_seconds()
        # else:
        #     subframe.loc[success_index, 'T1_' + from_colName + '-' + 'HO Attempt'] = 0

        # Computing T2: HO Attempt -- HO Success
        t2_df_Attempt = subframe[subframe['Event 5G-NR Events'] == 'NR Intrafreq Handover Attempt']
        t2_df_success = subframe[subframe['Event 5G-NR Events'] == 'NR Intrafreq Handover Success']
        # if t2_df_success.shape[0] > 0:
        subframe.loc[success_index, 'T2_HO Attempt-HO Success'] = (t2_df_success['TIME_STAMP'][t2_df_success.index[0]] - t2_df_Attempt['TIME_STAMP'][t2_df_Attempt.index[0]]).total_seconds()
        # else:
        #     subframe.loc[success_index, 'T2_HO Attempt-HO Success'] = 0
            # # Computing T3: HO Success -- RRCSetup Complete
        # for val in (subframe[['Event 5G-NR Events']].dropna()).values:
        #     if 'RRC Setup Success' in val[0]:
        #         break
        # t3_df_prach = subframe[subframe['Event 5G-NR Events'] == to_colName]
        # t3_df_rrcSetup = subframe[subframe['Event 5G-NR Events'] == val]
        # subframe.loc[success_index, 'T3_HO Success-RRCSetup Success'] = (t3_df_rrcSetup['TIME_STAMP'][t3_df_rrcSetup.index[0]] - t3_df_prach['TIME_STAMP'][t3_df_prach.index[0]]).total_seconds()
        # updated_df = df[df[colName].notna() & (df[colName] != 0)]

        # updated_dfs.append(df.loc[attempt_index:success_index])
        last_row = subframe.tail(1)
        updated_dfs.append(last_row)
        # updated_dfs.append(df.loc[[success_index]])
        # updated_dfs.append(updated_df)

    # Combine all updated DataFrames
    updated_df = pd.concat(updated_dfs)

    # Return only rows where 'Intra-NR HO Duration [Sec]' is non-zero and not NaN
    # return updated_df[updated_df['Intra-NR HO Duration [sec]'].notna() & (updated_df['Intra-NR HO Duration [sec]'] != 0)]
    return updated_df #[updated_df[colName].notna() & (updated_df[colName] != 0)]


