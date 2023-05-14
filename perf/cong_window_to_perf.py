"""
This program is an .ipybn notebook utility, used to convert special data.

"""

# %%
from __future__ import annotations
from json.tool import main
import re
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import math
import json
import yaml
import os
from collections import defaultdict
sns.set_theme(style="whitegrid")

# %%
'''Functions, that process files with multiple statlog.py runs outputs into data.'''
def convert_speed(fl):
    '''Converts big float speeds into 2^n format string'''
    if fl >= 1024**3//8:
        return f"{int(fl*8//(1024**3))} Gbit/s"
    if fl >= 1024**2//8:
        return f"{int(fl*8//(1024**2))} Mbit/s"
    if fl >= 1024//8:
        return f"{int(fl*8//(1024))} Kbit/s"

def convert_speed_to_mbit(fl):
    '''Converts big float speeds into Mbit format'''
    return int(fl*8//(1024**2))

def convert_speed_to_kbit(fl):
    '''Converts big float speeds into Kbit format'''
    return int(fl*8//1024)

def calculate_accuracy(th, pr):
    '''Calculates accuracy'''
    return math.fabs((th - pr) / th)

def get_data_yaml(paths, data_format_type):
    '''
    Each file in each dir is:

{additional_info: {channel_bw: 180, channel_congestion_control: bbrfrcst, channel_jitt: 1,
    channel_loss: 0.001, channel_rtt: 20, cong_window: 5000000}, content: [{'bytes sent: ': 23552551,
      'mean cwnd: ': 5000000.0, 'mean jitter: ': 0.238482, 'mean loss2: ': 0.951885,
      'mean loss: ': 31.848659, 'mean s_rtt: ': 51.163866, time: 0.0-1.0}, {'bytes sent: ': 23552551,
      'mean cwnd: ': 5000000.0, 'mean jitter: ': 0.238482, 'mean loss2: ': 0.951885,
      'mean loss: ': 31.848659, 'mean s_rtt: ': 51.163866, time: global}]}
    '''
    data = []
    good_data_amount = 0
    bad_data_amount = 0
    file_idx = 0
    for directory in paths:
        for filename in os.listdir(directory):
            file_idx += 1
            dir_file = os.path.join(directory, filename)
            if os.path.isfile(dir_file) and (filename != 'TooLongERRORs'):
                with open(dir_file, 'r') as f:
                    '''Analyse a single experiment'''
                    data_sample = []
                    datadict = yaml.safe_load(f)
                    '''Add parameters'''
                    conj_c = datadict['additional_info']['channel_congestion_control'].upper()
                    data_sample.append(conj_c)
                    theor_rtt = datadict['additional_info']['channel_rtt']
                    data_sample.append(theor_rtt)
                    theor_loss_percent = datadict['additional_info']['channel_loss']
                    data_sample.append(theor_loss_percent)
                    theor_speed_in_kbit = datadict['additional_info']['channel_bw'] * 1024
                    data_sample.append(theor_speed_in_kbit)
                    theor_jitt = datadict['additional_info']['channel_jitt']
                    data_sample.append(theor_jitt)
                    theor_cong_wind = int(datadict['additional_info']['cong_window'])
                    data_sample.append(theor_cong_wind)
                    parti = float(datadict['content'][0]['time'].split('-')[1]) - float(datadict['content'][0]['time'].split('-')[0])
                    '''The statlog.py puts global data in the end of the list => [-1]'''
                    real_speed_in_kbit = convert_speed_to_kbit(datadict['content'][-1]['bytes sent: '] / parti)
                    data_sample.append(real_speed_in_kbit)
                    real_rtt = datadict['content'][-1]['mean s_rtt: ']
                    data_sample.append(real_rtt)
                    real_loss1 = datadict['content'][-1]['mean loss: ']
                    data_sample.append(real_loss1)
                    real_loss2 = datadict['content'][-1]['mean loss2: ']
                    data_sample.append(real_loss2)
                    real_jitt = datadict['content'][-1]['mean jitter: ']
                    data_sample.append(real_jitt)
                    real_cong_wind = int(datadict['content'][-1]['mean cwnd: '])
                    data_sample.append(real_cong_wind)
                    content_length = len(datadict['content']) - 1 # Because 1 goes to calculated means
                    experiment_duration = float(content_length) * parti # In seconds
                    data_sample.append(experiment_duration)
                '''Skip if bad data (experiment was too fast). 
                Minimum experiment duration is currently 1min, accuracy is 1 second.'''
                if (experiment_duration < 0) or (real_loss1 > 2000):
                    print(filename)
                    bad_data_amount += 1
                    continue
                else:
                    good_data_amount += 1
                
                data.append(data_sample)
                if (file_idx % 100 == 0):
                    print(f"File number {file_idx} / ...")

    col = ["Congestion Controller", "Channel RTT (ms)", "Channel Loss (%)", "Channel BW (Kbit/s)",
           "Channel Jitter (ms)", "Preset Congestion Window (bytes)", "Sender Speed (Kbit/s)", 
           "Sender RTT (ms)", "Sender lost data to data inflight (%)", "Sender lost data to data sent (%)", 
           "Sender Jitter (ms)", "Sender Congestion Window (bytes)", "Experiment duration (sec)"]

    print("Good data samples amount: ", good_data_amount, " bad: ", bad_data_amount)
    good_data_amount = 1 if good_data_amount == 0 else good_data_amount
    return data, col, bad_data_amount / good_data_amount

def get_data_yaml_optimal(paths):
    '''
    Each file in each dir is:

[{channel_bw: 206, channel_congestion_control: bbrfrcst, channel_jitt: 1, channel_loss: 0.02,
    channel_rtt: 87, estimated_opt_cwnd: 1840239.0, estimated_opt_speed: 20097441.9414634,
    res_nfev: 13.0}, {channel_bw: 87, channel_congestion_control: bbrfrcst, channel_jitt: 1,

        ...

    channel_loss: 0.00125, channel_rtt: 33, estimated_opt_cwnd: 887011.0, estimated_opt_speed: 23245482.717073202,
    res_nfev: 12.0}]
    '''
    data = []
    for filename in paths:
        with open(filename, 'r') as f:
            datalist = yaml.safe_load(f)
            for datadict in datalist:
                '''Analyse a single experiment'''
                data_sample = []
                '''Add parameters'''
                conj_c = datadict['channel_congestion_control'].upper()
                data_sample.append(conj_c)
                theor_rtt = float(datadict['channel_rtt'])
                data_sample.append(theor_rtt)
                theor_loss_percent = float(datadict['channel_loss'])
                data_sample.append(theor_loss_percent)
                theor_speed_in_kbit = float(datadict['channel_bw'] * 1024)
                data_sample.append(theor_speed_in_kbit)
                theor_jitt = float(datadict['channel_jitt'])
                data_sample.append(theor_jitt)
                estimated_opt_speed = convert_speed_to_kbit(datadict['estimated_opt_speed'])
                data_sample.append(estimated_opt_speed)
                opt_cwnd = datadict['estimated_opt_cwnd']
                data_sample.append(opt_cwnd)
                res_nfev = datadict['res_nfev']
                data_sample.append(res_nfev)
        
                data.append(data_sample)

    col = ["Congestion Controller", "Channel RTT (ms)", "Channel Loss (%)", "Channel BW (Kbit/s)",
           "Channel Jitter (ms)", "Optimal Speed (Kbit/s)", "Optimal CWND (bytes)", "Estimation iterations number"]

    return pd.DataFrame(data, columns=col)

def get_data_for_heatmap(paths):
    '''(Legacy code of year 2022)
    Each file in each dir is:

{additional_info: {channel_bw: 200, channel_congestion_control: bbrfrcst, channel_jitt: 1,
    channel_loss: 2, channel_rtt: 90, cwnd: 2124975.157509446}, content: [{'bytes sent: ': 5116144,

        ...

      'mean cwnd: ': 1297416.576883, 'mean jitter: ': 0.378005, 'mean loss2: ': 0.004307,
      'mean loss: ': 0.416563, 'mean s_rtt: ': 94.821238, 'state: ': ExitsFORECAST,
      time: global}]}
    '''
    sla_d = defaultdict(float)
    anno_d = defaultdict(dict)
    for filename in paths:
        with open(filename, 'r') as f:
            datalist = yaml.safe_load(f)
            for datadict in datalist:
                p_rtt = datadict['additional_info']['channel_rtt']
                p_loss = datadict['additional_info']['channel_loss']
                p_bw = datadict['additional_info']['channel_bw'] * 1024
                parti = float(datadict['content'][0]['time'].split('-')[1]) - float(datadict['content'][0]['time'].split('-')[0])
                real_bw = convert_speed_to_kbit(datadict['content'][-1]['bytes sent: '] / parti)
                if "samples" in anno_d[(p_rtt, p_bw)]:
                    '''Number of experiments'''
                    anno_d[(p_rtt, p_bw)]["samples"] += 1
                else:
                    anno_d[(p_rtt, p_bw)]["samples"] = 1
                if "loss" in anno_d[(p_rtt, p_bw)]:
                    '''It is mean loss set in channel (by experiments)'''
                    anno_d[(p_rtt, p_bw)]["p_loss"] = anno_d[(p_rtt, p_bw)]["p_loss"] * (anno_d[(p_rtt, p_bw)]["samples"] - 1) / anno_d[(p_rtt, p_bw)]["samples"]
                    anno_d[(p_rtt, p_bw)]["p_loss"] += p_loss / anno_d[(p_rtt, p_bw)]["samples"]
                else:
                    anno_d[(p_rtt, p_bw)]["p_loss"] = p_loss
                if "real_bw" in anno_d[(p_rtt, p_bw)]:
                    '''It is mean real bw (by experiments)'''
                    anno_d[(p_rtt, p_bw)]["real_bw"] = anno_d[(p_rtt, p_bw)]["real_bw"] * (anno_d[(p_rtt, p_bw)]["samples"] - 1) / anno_d[(p_rtt, p_bw)]["samples"]
                    anno_d[(p_rtt, p_bw)]["real_bw"] += real_bw / anno_d[(p_rtt, p_bw)]["samples"]
                else:
                    anno_d[(p_rtt, p_bw)]["real_bw"] = real_bw
                if "min_real_bw" in anno_d[(p_rtt, p_bw)]:
                    '''It is min real bw (by experiments)'''
                    if real_bw < anno_d[(p_rtt, p_bw)]["min_real_bw"]:
                        anno_d[(p_rtt, p_bw)]["min_real_bw"] = real_bw
                else:
                    anno_d[(p_rtt, p_bw)]["min_real_bw"] = real_bw
                if "max_real_bw" in anno_d[(p_rtt, p_bw)]:
                    '''It is max real bw (by experiments)'''
                    if real_bw > anno_d[(p_rtt, p_bw)]["max_real_bw"]:
                        anno_d[(p_rtt, p_bw)]["max_real_bw"] = real_bw
                else:
                    anno_d[(p_rtt, p_bw)]["max_real_bw"] = real_bw
                # print(anno_d[(p_rtt, p_bw)]["samples"])
                sla_d[(p_rtt, p_bw)] = sla_d[(p_rtt, p_bw)] * (anno_d[(p_rtt, p_bw)]["samples"] - 1) / anno_d[(p_rtt, p_bw)]["samples"]
                if i[3] == "SLA IS OKAY":
                    sla_d[(p_rtt, p_bw)] += 1 / anno_d[(p_rtt, p_bw)]["samples"]
                anno_d[(p_rtt, p_bw)]["annotation4"] = "SLA: {}/{}\nChannel loss: {}\nBBRFRCST speed: {}".format(
                    int(sla_d[(p_rtt, p_bw)] * anno_d[(p_rtt, p_bw)]["samples"]),
                    anno_d[(p_rtt, p_bw)]["samples"],
                    anno_d[(p_rtt, p_bw)]["p_loss"],
                    convert_speed(anno_d[(p_rtt, p_bw)]["real_bw"]),
                )
                anno_d[(p_rtt, p_bw)]["annotation3"] = "SLA: {}/{}\nChannel loss: {}".format(
                    int(sla_d[(p_rtt, p_bw)] * anno_d[(p_rtt, p_bw)]["samples"]),
                    anno_d[(p_rtt, p_bw)]["samples"],
                    anno_d[(p_rtt, p_bw)]["p_loss"],
                )
    patt = re.compile(r"BBR2experiment. (.*)(\n|.)*?Mean speed (.*)\n")
    for i in patt.findall(maintext):
        # print(i)
        p_rtt = float(i[0].split()[1])
        p_loss = float(i[0].split()[3])
        p_bw = float(i[0].split()[5])
        real_bbr_bw = float(i[2].split()[0]) / float(i[2].split()[3])
        '''Next condition can be ignored'''
        if (p_rtt, p_bw) not in anno_d:
            continue
        if "bbr_samples" in anno_d[(p_rtt, p_bw)]:
            '''Number of experiments'''
            anno_d[(p_rtt, p_bw)]["bbr_samples"] += 1
        else:
            anno_d[(p_rtt, p_bw)]["bbr_samples"] = 0
        if "bbr_loss" in anno_d[(p_rtt, p_bw)]:
            '''It is mean loss set in channel (by experiments)'''
            anno_d[(p_rtt, p_bw)]["bbr_p_loss"] = anno_d[(p_rtt, p_bw)]["bbr_p_loss"] * (anno_d[(p_rtt, p_bw)]["bbr_samples"] - 1) / anno_d[(p_rtt, p_bw)]["bbr_samples"]
            anno_d[(p_rtt, p_bw)]["bbr_p_loss"] += p_loss / anno_d[(p_rtt, p_bw)]["bbr_samples"]
        else:
            anno_d[(p_rtt, p_bw)]["bbr_p_loss"] = p_loss
        if "bbr_real_bw" in anno_d[(p_rtt, p_bw)]:
            '''It is mean real bw (by experiments)'''
            anno_d[(p_rtt, p_bw)]["bbr_real_bw"] = anno_d[(p_rtt, p_bw)]["bbr_real_bw"] * (anno_d[(p_rtt, p_bw)]["bbr_samples"] - 1) / anno_d[(p_rtt, p_bw)]["bbr_samples"]
            anno_d[(p_rtt, p_bw)]["bbr_real_bw"] += real_bbr_bw / anno_d[(p_rtt, p_bw)]["bbr_samples"]
        else:
            anno_d[(p_rtt, p_bw)]["bbr_real_bw"] = real_bbr_bw
        if "bbr_min_real_bw" in anno_d[(p_rtt, p_bw)]:
            '''It is min real bw (by experiments)'''
            if real_bbr_bw < anno_d[(p_rtt, p_bw)]["bbr_min_real_bw"]:
                anno_d[(p_rtt, p_bw)]["bbr_min_real_bw"] = real_bbr_bw
        else:
            anno_d[(p_rtt, p_bw)]["bbr_min_real_bw"] = real_bbr_bw
        if "bbr_max_real_bw" in anno_d[(p_rtt, p_bw)]:
            '''It is max real bw (by experiments)'''
            if real_bbr_bw > anno_d[(p_rtt, p_bw)]["bbr_max_real_bw"]:
                anno_d[(p_rtt, p_bw)]["bbr_max_real_bw"] = real_bbr_bw
        else:
            anno_d[(p_rtt, p_bw)]["bbr_max_real_bw"] = real_bbr_bw
        anno_d[(p_rtt, p_bw)]["annotation1"] = "SLA: {}/{}\nChannel loss: {}\nBBRFRCST speed: {}\nCUBIC speed:     {}".format(
            int(sla_d[(p_rtt, p_bw)] * anno_d[(p_rtt, p_bw)]["samples"]),
            anno_d[(p_rtt, p_bw)]["samples"],
            anno_d[(p_rtt, p_bw)]["p_loss"],
            convert_speed(anno_d[(p_rtt, p_bw)]["real_bw"]),
            convert_speed(anno_d[(p_rtt, p_bw)]["bbr_real_bw"]),
        )
        anno_d[(p_rtt, p_bw)]["annotation2"] = "SLA: {}/{}\nChannel loss: {}\nBBRFRCST speed: {}\n(min: {}, max: {})\nBBR2 speed:         {}\n(min: {}, max: {})".format(
            int(sla_d[(p_rtt, p_bw)] * anno_d[(p_rtt, p_bw)]["samples"]),
            anno_d[(p_rtt, p_bw)]["samples"],
            anno_d[(p_rtt, p_bw)]["p_loss"],
            convert_speed(anno_d[(p_rtt, p_bw)]["real_bw"]),
            convert_speed(anno_d[(p_rtt, p_bw)]["min_real_bw"]),
            convert_speed(anno_d[(p_rtt, p_bw)]["max_real_bw"]),
            convert_speed(anno_d[(p_rtt, p_bw)]["bbr_real_bw"]),
            convert_speed(anno_d[(p_rtt, p_bw)]["bbr_min_real_bw"]),
            convert_speed(anno_d[(p_rtt, p_bw)]["bbr_max_real_bw"]),
        )
        # print(anno_d[(p_rtt, p_bw)]["annotation"])
    return sla_d, anno_d

# %%
'''Define pathes'''

'''Path to save csv'''
path_save_csv = 'csv_stat'

'''data_format_type equals following numbers depending on the task:
1. Speed calculation on congestion window;
3. Optimal congestion window calculation on channel parameters;
'''
data_format_type = 2
if data_format_type == 1:
    paths_yml_dir = ['perfres_w3_speed_v3']
if data_format_type == 2:
    paths_yml_dir = ['perfresCWNDWed_Mar__1_12.35.15_2023.optimal',
                     'perfresCWNDWed_Mar__1_12.29.49_2023.optimal',
                     'perfresCWND_additional_dataset/small_rtt/perfresCWNDTue_May__2_09.41.42_2023.optimal',
                     'perfresCWND_additional_dataset/small_rtt/perfresCWNDTue_May__2_23.39.11_2023.optimal',
                     'perfresCWND_additional_dataset/big_rtt/perfresCWNDSun_Apr_16_16.27.55_2023.optimal']

'''Path to directory with xlsx files'''
path_savexlsxs = "cc_perf_tests"

# %%
'''Put files needed to be composed into a dataframe in statfiles, then specify algos, that we want to analyse.'''
if __name__ == "__main__":
    if data_format_type in [1]:
        data_lists, col, bad_data_fraction1 = get_data_yaml(paths_yml_dir, data_format_type=data_format_type)
        df = pd.DataFrame(data_lists, columns=col)
    if data_format_type in [2]:
        data = get_data_yaml_optimal(paths_yml_dir)
    else: 
        print(f"Current task/format is {data_format_type} => skipping")

# %%
'''If we want to group by something and calculate means'''
if __name__ == "__main__":
    print(data)
# data.loc[data["Optimal CWND (bytes)"] < 3300000].loc[data["Optimal CWND (bytes)"] > 30000].sort_values(by="Optimal CWND (bytes)").head(20)
    data = data.loc[data["Optimal CWND (bytes)"] < 3300000].loc[data["Optimal CWND (bytes)"] > 30000].sort_values(by="Optimal CWND (bytes)") # Treshold encountered

# %%
def explore_cong_window_in_one_dot(df_dot, dot, channel_features):
    fig, axes = plt.subplots(4, 1, figsize=(9, 13))
    fig.suptitle(f'Dependencies between BBR Congestion Window and features.\nRTT = {dot[1]} ms, Loss = {dot[2]} %, BW = {dot[3]} Kbit/s.')
    sns.lineplot(ax=axes[0],data=df_dot.loc[df["Preset Congestion Window (bytes)"] < 2000000], x="Preset Congestion Window (bytes)", y=f"Sender Speed (Kbit/s)", ci=70)
    # sns.lineplot(ax=axes[1],data=df_dot, x="Preset Congestion Window (bytes)", y=f"Sender lost data to data inflight (%)", ci=70)
    sns.lineplot(ax=axes[1],data=df_dot.loc[df["Preset Congestion Window (bytes)"] < 2000000], x="Preset Congestion Window (bytes)", y=f"Sender lost data to data sent (%)", ci=70)
    sns.lineplot(ax=axes[2],data=df_dot.loc[df["Preset Congestion Window (bytes)"] < 2000000], x="Preset Congestion Window (bytes)", y=f"Sender RTT (ms)", ci=70)
    # sns.lineplot(ax=axes[3],data=df_dot.loc[df["Preset Congestion Window (bytes)"] < 2000000], x="Preset Congestion Window (bytes)", y=f"Experiment duration (sec)", ci=70)

if __name__ == "__main__":
    if data_format_type in [1]:
        channel_features = ["Congestion Controller", "Channel RTT (ms)", 'Channel Loss (%)', "Channel BW (Kbit/s)", "Channel Jitter (ms)"]
        dots = list(df[channel_features].value_counts().index)
        tmp = 0
        for dot in dots:
            df_dot = df
            tmp += 1
            for feature in channel_features:
                df_dot = df_dot.loc[(df_dot[feature] == dot[channel_features.index(feature)])]
            if tmp >= 0:
                print("Dot:", df_dot, "Shape:", df_dot.shape)
                explore_cong_window_in_one_dot(df_dot, dot, channel_features)
    else: 
        print(f"Current task/format is {data_format_type} => skipping")

# %%
'''Put results in excel'''
import openpyxl

if __name__ == "__main__":
    '''Save results in a xlsx'''
    if data_format_type in [2]:
        data.to_excel(f"{path_savexlsxs}/optimal_speed_ML_additional_dataset_v4.xlsx")
    else:
        print(f"Current task/format is {data_format_type} => skipping")

# %%
'''Correlation examination 1'''
if __name__ == "__main__":
    plt.figure(figsize=(10, 8))
    data_tmp = data
    data_tmp['BDP'] = data_tmp['Channel BW (Kbit/s)'] * data_tmp['Channel RTT (ms)']
    sns.heatmap(data_tmp.corr(method='spearman'), annot=True)

    data_tmp['ratio'] = data_tmp['Optimal CWND (bytes)'] / data_tmp['BDP']
    data_tmp.describe()

# %%
data_tmp.loc[data_tmp['ratio'] < 0.1]

# %%
