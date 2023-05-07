"""
This program is an .ipybn notebook utility.

"""

#%%
from json.tool import main
import re
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import yaml
import os
from collections import defaultdict

#%%
def convert_speed(fl):
    '''Converts big float speeds into 2^n format string'''
    if fl >= 1024**3//8:
        return f"{int(fl*8//(1024**3))} Gbit/s"
    if fl >= 1024**2//8:
        return f"{int(fl*8//(1024**2))} Mbit/s"
    if fl >= 1024//8:
        return f"{int(fl*8//(1024))} Kbit/s"

def convert_speed_to_kbit(fl):
    '''Converts big float speeds into Kbit format'''
    return int(fl*8//1024)

def get_data_for_heatmap(paths, main_cc="bbrfrcst", additional_cc="bbr2"):
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
    file_idx = 0
    good_data_amount = 0
    bad_data_amount = 0
    for directory in paths:
        for filename in os.listdir(directory):
            file_idx += 1
            dir_file = os.path.join(directory, filename)
            if os.path.isfile(dir_file) and (filename not in ['TooLongERRORs', 'OtherERRORs']):
                with open(dir_file, 'r') as f:
                    '''Analyse a single experiment'''
                    datadict = yaml.safe_load(f)
                    if not isinstance(datadict, dict):
                        bad_data_amount += 1
                        continue

                    '''Add parameters'''
                    p_rtt = datadict['additional_info']['channel_rtt']
                    p_loss = datadict['additional_info']['channel_loss']
                    p_bw = datadict['additional_info']['channel_bw']

                    # if p_loss != 2:
                    #     file_idx -= 1
                    #     continue

                    parti = float(datadict['content'][0]['time'].split('-')[1]) - float(datadict['content'][0]['time'].split('-')[0])
                    
                    if (datadict['additional_info']['channel_congestion_control'] == main_cc):
                        real_bw = convert_speed_to_kbit(datadict['content'][-1]['bytes sent: '] / parti)
                        real_rtt = datadict['content'][-1]['mean s_rtt: ']
                        real_loss = datadict['content'][-1]['mean loss: ']

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
                        if "rtt" in anno_d[(p_rtt, p_bw)]:
                            '''It is mean real bw (by experiments)'''
                            anno_d[(p_rtt, p_bw)]["real_rtt"] = anno_d[(p_rtt, p_bw)]["real_rtt"] * (anno_d[(p_rtt, p_bw)]["samples"] - 1) / anno_d[(p_rtt, p_bw)]["samples"]
                            anno_d[(p_rtt, p_bw)]["real_rtt"] += real_rtt / anno_d[(p_rtt, p_bw)]["samples"]
                        else:
                            anno_d[(p_rtt, p_bw)]["real_rtt"] = real_rtt
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
                        if datadict['content'][-1]['state: '] == "SLA is OK":
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
                        anno_d[(p_rtt, p_bw)]["rtt bbrfrcst annotation"] = "SLA RTT: {}\nBBRFRCST RTT: {}".format(
                            p_rtt,
                            anno_d[(p_rtt, p_bw)]["real_rtt"],
                        )
                    elif (datadict['additional_info']['channel_congestion_control'] == additional_cc):
                        real_bbr_bw = convert_speed_to_kbit(datadict['content'][-1]['bytes sent: '] / parti)
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
                        anno_d[(p_rtt, p_bw)]["annotation1"] = "SLA: {}/{}\nChannel loss: {}\nBBRFRCST speed: {}\n{} speed:     {}".format(
                            int(sla_d[(p_rtt, p_bw)] * anno_d[(p_rtt, p_bw)]["samples"]),
                            anno_d[(p_rtt, p_bw)]["samples"],
                            anno_d[(p_rtt, p_bw)]["p_loss"],
                            convert_speed(anno_d[(p_rtt, p_bw)]["real_bw"]),
                            additional_cc.upper(),
                            convert_speed(anno_d[(p_rtt, p_bw)]["bbr_real_bw"]),
                        )
                        anno_d[(p_rtt, p_bw)]["annotation2"] = "SLA: {}/{}\nChannel loss: {}\nBBRFRCST speed: {}\n(min: {}, max: {})\n{} speed:         {}\n(min: {}, max: {})".format(
                            int(sla_d[(p_rtt, p_bw)] * anno_d[(p_rtt, p_bw)]["samples"]),
                            anno_d[(p_rtt, p_bw)]["samples"],
                            anno_d[(p_rtt, p_bw)]["p_loss"],
                            convert_speed(anno_d[(p_rtt, p_bw)]["real_bw"]),
                            convert_speed(anno_d[(p_rtt, p_bw)]["min_real_bw"]),
                            convert_speed(anno_d[(p_rtt, p_bw)]["max_real_bw"]),
                            additional_cc.upper(),
                            convert_speed(anno_d[(p_rtt, p_bw)]["bbr_real_bw"]),
                            convert_speed(anno_d[(p_rtt, p_bw)]["bbr_min_real_bw"]),
                            convert_speed(anno_d[(p_rtt, p_bw)]["bbr_max_real_bw"]),
                        )
                        # print(anno_d[(p_rtt, p_bw)]["annotation"])
    good_data_amount = file_idx
    return sla_d, anno_d, good_data_amount, bad_data_amount


#%%
# INPUT:
statfiles = ["sla_check_diploma/sla_p12_beta_func_additional_dataset_catboost_less_ultra",
             "sla_check_diploma/sla_bbr2_p2"]

additional_cc = "bbr2"
sla, anno, _, _ = get_data_for_heatmap(statfiles, main_cc="bbrfrcst", additional_cc=additional_cc)


# %%

task_names_possible = [
    "annotation1", # Compare speed of BBRFORECAST and additional_cc
    "annotation",
    "annotation3" # Compare RTT of BBRFORECAST and additional_cc
]
task_names = [
    "annotation1"
]

for task_name in task_names:
    rtts = set()
    bws = set()
    for i in sla.keys():
        rtts.add(i[0])
        bws.add(i[1])
    l_rtts = sorted(list(rtts))[::-1]
    l_bws = sorted(list(bws))
    pd_sla = []
    pd_anno = []
    for i in l_rtts:
        tmp_l = []
        tmp_a_l = []
        for j in l_bws:
            if task_name == "annotation3":
                tmp_l.append(sla[(i, j)])
                if "annotation3" in anno[(i, j)]:
                    tmp_l[-1] += 1
                    tmp_a_l.append(anno[(i, j)]["annotation3"])
                else:
                    tmp_a_l.append("No data")
            if task_name == "annotation1":
                if task_name in anno[(i, j)]:# and sla[(i, j)] == 1:
                    tmp_var = anno[(i, j)]["real_bw"] / anno[(i, j)]["bbr_real_bw"]
                    # if tmp_var < 1:
                        # tmp_var *= 4
                        # tmp_var -= 4
                    tmp_l.append(tmp_var)
                    tmp_a_l.append(anno[(i, j)][task_name])
                # elif "annotation1" in anno[(i, j)] and sla[(i, j)] != 1:
                    # tmp_l.append(0)
                    # tmp_a_l.append("SLA is not full")
                else:
                    tmp_l.append(0)
                    tmp_a_l.append("No data")
        if (tmp_l):
            pd_anno.append(tmp_a_l)
            pd_sla.append(tmp_l)
    pd2_sla = pd.DataFrame(pd_sla, columns = l_bws, index = l_rtts)
    pd2_anno = pd.DataFrame(pd_anno)
    # print(pd_anno, pd_sla, np.array(pd_sla).size, np.array(pd_anno).size)
    # pd_anno = pd.Series(anno).reset_index()
    if task_name == "annotation3":
        plt.figure(figsize=(18, 12))
        sns.heatmap(pd2_sla, annot=pd2_anno, fmt="", center=0, linewidths=.5)
        plt.title('SLA Goodness Statistics: FrcstR = 120 Mbit/s, FrcstRTT = 100 ms, FrcstLoss = 0.1')
    if task_name == "annotation1":
        plt.figure(figsize=(20, 11))
        cmap = sns.diverging_palette(220, 10, as_cmap=True, s=100).reversed()
        sns.heatmap(pd2_sla, annot=pd2_anno, fmt="", center=1, linewidths=.5, robust=True, cmap=cmap)
        plt.title(f'BBRFrcst vs {additional_cc.upper()} Statistics')
    plt.xlabel('Channel BW - FrcstBW')
    plt.ylabel('Channel RTT - FrcstRTT')


# %%
