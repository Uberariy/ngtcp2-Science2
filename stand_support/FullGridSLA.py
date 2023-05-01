'''
This script runs a set of experiments to collect dataset of the best CWND parameters by channel parameters.
Uses "inopsy-stand" as a stand and "ngtcp2-Science" as an agent.
'''

import sys
import os
import time
import math
import numpy as np
import random
import signal
import yaml
import subprocess
from scipy.optimize import Bounds, minimize, LinearConstraint

from pathlib import Path
from RunAllExperiments import curr_time_to_path, TooLongExperiment, signal_handler
from ChangeExperiment import change_experiment
from predict_cwnd import load_and_predict

if __name__ == "__main__":
    '''Change file paths below'''
    path_to_experiment = Path("experimentData.yml")
    path_to_change = Path("ChangeExperiment.py")
    path_to_mininet_run = Path("main.py")
    path_to_agent = Path("../ngtcp2-Science/")
    path_to_statlog_folder_in_agent = Path("perf/sla_bbr2_p1/")

    '''Ensure, that folder to save experiments exists'''
    if not os.path.exists(path_to_agent / path_to_statlog_folder_in_agent):
        os.makedirs(path_to_agent / path_to_statlog_folder_in_agent)

    '''CONFIG: Change parameters grid below'''
    selection_rtts = [20, 50, 100] #[10, 20, 40, 50, 75, 100]
    selection_bws = [40, 100, 200, 240] #[40, 80, 120, 160, 200, 240]
    selection_losses = [0.001, 0.1, 2] # [0.001, 0.01, 0.1, 0.5, 1, 2, 4] # [0.002, 0.005, 0.075, 0.5, 0.00125, 0.0015, 0.01, 0.015, 0.02, 0.1, 0.2, 0.4, 0.05, 0.15, 0.0025, 0.0075, 0.025, 0.03, 0.04, 0.003, 0.004, 0.0125, 0.3, 0.001, 0.75, 0.25, 3, 1.5, 2, 1, 1.25, 2.5]
    selection_jitts = [1] # Set jitter to 1ms, because w3 is not patched
    launches_per_parameter_set = 3
    
    expC = launches_per_parameter_set * len(selection_bws) * len(selection_jitts) * len(selection_losses) * len(selection_rtts) # Manual calculation of experiment upper bound
    timeConstraint = 40 # If something goes wrong during experiment
    minimal_experiment_length = 20

    expN = 0
    cur_jitt = selection_jitts[0]
    cc = "bbr2"
    t_start = time.time()

    for cur_rtt in selection_rtts:
        for cur_bw in selection_bws:
            for cur_loss in selection_losses:

                signal.signal(signal.SIGALRM, signal_handler)
                expN_aim = expN + launches_per_parameter_set
                while expN < expN_aim:
                    expN += 1
                    signal.alarm(timeConstraint)
                    try:
                        t_one_start = time.time()
                        print(f"#### MinimizeBBRParamTest: Running experiment ({expN}/...) - rtt: {cur_rtt} loss: {cur_loss} bw: {cur_bw} jitt: {cur_jitt}\t... ")
                        cwnd = load_and_predict(path_to_models='cwnd_models/', rtt=cur_rtt, loss=cur_loss, bw=1024*cur_bw) # Convert to Kbit/s
                        print(f"#### CWND got from regressor: {cwnd}\t ")

                        '''Assosiate statfile name with current second and number (optional)
                        (Assume, that no more than one exp during a second is held)'''
                        statfile = path_to_statlog_folder_in_agent / f"{expN}_exp_{curr_time_to_path()}"

                        '''Change some of the experiment parameters, others will stay constant'''
                        change_experiment(exp_path=str(path_to_experiment), p_rtt=cur_rtt, p_loss=cur_loss, p_bw=cur_bw, p_jitt=cur_jitt, cc=cc, r_path=str(statfile), cong_window=cwnd)

                        '''Add additional information '''
                        info = dict()
                        info['channel_rtt'] = cur_rtt
                        info['channel_loss'] = cur_loss
                        info['channel_bw'] = cur_bw
                        info['channel_jitt'] = cur_jitt
                        info['channel_congestion_control'] = cc
                        info['cwnd'] = float(cwnd)
                        savedict = {"additional_info": info}

                        '''Run stand with modified experiment data'''
                        p_topo = subprocess.Popen(f"sudo /usr/bin/python3 {path_to_mininet_run} {path_to_experiment} --single", shell = True)
                        p_topo.wait()
                        t_one_end = time.time()
                        print(f"#### RunAll: Experiment finished. Time taken: {t_one_end - t_one_start} sec. ({(t_one_end - t_one_start) / 60} minutes)")
            
                        with open(path_to_agent / statfile, "r") as f:
                            datalist = yaml.safe_load(f)
                        
                        '''Real speed estimation for mimimization process'''
                        experiment_duration = float(datalist[-2]['time'].split('-')[1])
                        if experiment_duration < minimal_experiment_length:
                            expN -= 1

                            print(f"#### Experiment is too short: {experiment_duration} seconds")
                        else:
                            '''The experiment list is not too long for this operation'''
                            savedict["content"] = datalist
                            with open(path_to_agent / statfile, 'w') as f:
                                yaml.dump(savedict, f, default_flow_style=True)
                    except TooLongExperiment as E:
                        with open(path_to_agent / path_to_statlog_folder_in_agent / "TooLongERRORs", 'a+') as f:
                            f.write(' '.join([i + ":" + str(j) for i, j in info.items()]) + f' {expN}{curr_time_to_path}\n')
                        '''Kill mininet if it lags'''
                        # p_topo.kill() -- Not working!
                        for line in os.popen("ps ax | grep " + f"'/usr/bin/python3 {path_to_mininet_run} {path_to_experiment} --single'" + " | grep -v grep"):
                            pid = int(line.split()[0])
                            print("PID ", pid, " killed!")
                            os.kill(pid, signal.SIGKILL)
                        print(E)
                        signal.alarm(0)
                        expN -= 1 # Hold this experiment again
                        continue
                    except KeyboardInterrupt as E:
                        print('\nKeyboardInterrupt => Returning...')
                        signal.alarm(0)
                        continue
                    except Exception as E:
                        print(f'\nUnexpected exception happened during experiment:\n{E}')
                        with open(path_to_agent / path_to_statlog_folder_in_agent / "OtherERRORs", 'a+') as f:
                            f.write(' '.join([i + ":" + str(j) for i, j in info.items()]) + f' {expN}{curr_time_to_path}\t')
                            f.write(f'Error is {E}\n')
                        expN -= 1
                    signal.alarm(0)
                    # if (expN >= ...):
                    #     exit()
                    
    t_end = time.time()
    print(f"#### Total time taken: {t_end - t_start} sec. ({(t_end - t_start) / 60} minutes)")
