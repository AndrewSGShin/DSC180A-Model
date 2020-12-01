### SETUP
# Import the necessary libraries.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from scipy.signal import find_peaks, peak_widths
# from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import os, sys

### HELPER FUNCTIONS
# Functions for received packets.
def receive_psizes(row):
    output = []
    for i in range(len(row.packet_dirs)):
        if row.packet_dirs[i] == '2':
            output.append(row.packet_sizes[i])
    return output

def receive_ptimes(row):
    output = []
    for i in range(len(row.packet_dirs)):
        if row.packet_dirs[i] == '2':
            output.append(row.packet_times[i])
    return output

# Functions for sent packets.
def send_psizes(row):
    output = []
    for i in range(len(row.packet_dirs)):
        if row.packet_dirs[i] == '1':
            output.append(row.packet_sizes[i])
    return output

def send_ptimes(row):
    output = []
    for i in range(len(row.packet_dirs)):
        if row.packet_dirs[i] == '1':
            output.append(row.packet_times[i])
    return output

### ALL-IN-ONE DATASET CLEANING FUNCTION
def etl_dataset(csv, upload_threshold=100000, download_threshold=1000000, height=0.5):
    # Read in the data.
    data = pd.read_csv(csv)
    # Do basic cleaning/transformations.
#     data['Time'] = pd.to_datetime(data['Time'], unit='s')
    data['packet_times'] = data['packet_times'].apply(lambda x: x.split(';'))
    data['packet_sizes'] = data['packet_sizes'].apply(lambda x: x.split(';'))
    data['packet_dirs'] = data['packet_dirs'].apply(lambda x: x.split(';'))
    data['1->2Pkt_Times'] = data.apply(send_ptimes, axis=1)
    data['2->1Pkt_Times'] = data.apply(receive_ptimes, axis=1)
    data['1->2Pkt_Sizes'] = data.apply(send_psizes, axis=1)
    data['2->1Pkt_Sizes'] = data.apply(receive_psizes, axis=1)
    # Prepare for feature engineering.
    # Find upload/download/all peak indices.
    upload_peaks = find_peaks(data['1->2Bytes'], height=upload_threshold)[0]
    download_peaks = find_peaks(data['2->1Bytes'], height=download_threshold)[0]
    # Functions for peak locations.
    def upload_peak(row):
        if row.name in list(upload_peaks):
            return 1
        return 0

    def download_peak(row):
        if row.name in list(download_peaks):
            return 1
        return 0

    # Functions to help derive peak widths.
    def upload_peak_width(row):
        if row.name in list(upload_peaks):
            return upload_widths[list(upload_peaks).index(row.name)]
        return 0

    def download_peak_width(row):
        if row.name in list(download_peaks):
            return download_widths[list(download_peaks).index(row.name)]
        return 0
    
    data['1->2_is_peak'] = data.apply(upload_peak, axis=1)
    data['2->1_is_peak'] = data.apply(download_peak, axis=1)
    # Find upload/download peak widths.
    upload_widths = peak_widths(data['1->2Bytes'], upload_peaks, rel_height=height)[0]
    download_widths = peak_widths(data['2->1Bytes'], download_peaks, rel_height=height)[0]
    data['1->2_peak_width'] = data.apply(upload_peak_width, axis=1)
    data['2->1_peak_width'] = data.apply(download_peak_width, axis=1)
    return data

### ENGINEER FEATURES
def feature_engineering(data, streaming=None, download_threshold=1000000):
    output = []
    # Feature 1: proportion of peaks/2min capture
    percent = data['2->1_is_peak'].sum() / len(data['2->1_is_peak'])
    # Feature 2: standard deviation of time intervals between peaks/2min capture
    x = data['Time']
    y = data['2->1Bytes']
    peaks, _ = find_peaks(y, height=download_threshold)
    time_std = np.std(x[peaks].diff().values[1:])
    # Feature 3: standard deviation of peak heights/2min capture
    height_std = np.std(y[peaks])
    # Feature 4: standard deviation of the downloaded packets' sizes
    pck_sizes = [int(x) for x in data['2->1Pkt_Sizes'].sum()]
    size_std = np.std(pck_sizes)
    # Feature 5: mean of the downloaded packets' sizes
#     size_mean = np.mean(pck_sizes)
    # Feature 6: ratio of large received packets to all received packets
    pkts_received = pd.DataFrame({'2->1Pkt_Times': data['2->1Pkt_Times'].sum(), '2->1Pkt_Sizes': data['2->1Pkt_Sizes'].sum()}).astype(int)
    large_ratio = pkts_received[pkts_received['2->1Pkt_Sizes'] > 1200]['2->1Pkt_Sizes'].count() / pkts_received['2->1Pkt_Sizes'].count()
    # Feature 7: ratio of small sent packets to all sent packets
    pkts_sent = pd.DataFrame({'1->2Pkt_Times': data['1->2Pkt_Times'].sum(), '1->2Pkt_Sizes': data['1->2Pkt_Sizes'].sum()}).astype(int)
    small_ratio = pkts_sent[pkts_sent['1->2Pkt_Sizes'] < 200]['1->2Pkt_Sizes'].count() / pkts_sent['1->2Pkt_Sizes'].count()
    
    if streaming is not None:
        return [percent, time_std, height_std, size_std, large_ratio, small_ratio, int(streaming)]
    else:
        return [percent, time_std, height_std, size_std, large_ratio, small_ratio]

### BUILD FEATURE TABLE
def build_features(datasets, is_streaming=None, upload_threshold=100000, download_threshold=1000000, height=0.5):
    if is_streaming is not None:
        cols = ['is_peak%', 'peak_interval_std', 'peak_height_std', 'pckt_size_std', 'large_pkt_received_ratio', 'small_pkt_sent_ratio', 'is_streaming']
        if type(datasets) == str:
            temp = etl_dataset(csv=datasets, upload_threshold=upload_threshold, download_threshold=download_threshold, height=height)
            row = feature_engineering(temp, streaming=is_streaming, download_threshold=download_threshold)
            return pd.DataFrame([row], columns=cols).fillna(0)
        else:
            rows = []
            count = 0
            for i in datasets:
                temp = etl_dataset(csv=i, upload_threshold=upload_threshold, download_threshold=download_threshold, height=height)
                rows.append(feature_engineering(temp, streaming=is_streaming[count], download_threshold=download_threshold))
                count += 1
            return pd.DataFrame(rows, columns=cols).fillna(0)
    else:
        cols = ['is_peak%', 'peak_interval_std', 'peak_height_std', 'pckt_size_std', 'large_pkt_received_ratio', 'small_pkt_sent_ratio']
        if type(datasets) == str:
            temp = etl_dataset(csv=datasets, upload_threshold=upload_threshold, download_threshold=download_threshold, height=height)
            row = feature_engineering(temp, download_threshold=download_threshold)
            return pd.DataFrame([row], columns=cols).fillna(0)
        else:
            rows = []
            count = 0
            for i in datasets:
                temp = etl_dataset(csv=i, upload_threshold=upload_threshold, download_threshold=download_threshold, height=height)
                rows.append(feature_engineering(temp, download_threshold=download_threshold))
                count += 1
            return pd.DataFrame(rows, columns=cols).fillna(0)
    
def run_model(test, is_streaming=None):
    train = []
    for i in range(25):
        train.append(build_features('training data/video/' + os.listdir('training data/video/')[i], True))
        train.append(build_features('training data/no-video/' + os.listdir('training data/no-video/')[i], False))
    train = pd.concat(train, ignore_index=True)
    train_X = train.iloc[:, :-1]
    train_y = train.iloc[:, -1].values
    
    testing = build_features(test)
    
    # Logistic Regression Model
    rf = RandomForestClassifier()
    rf.fit(train_X, train_y)
    return rf.predict(testing)

filepath = sys.argv[1]
print('The predicted classification for the input is: ', run_model(filepath))