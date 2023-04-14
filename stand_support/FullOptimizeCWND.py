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

if __name__ == "__main__":
    '''Change file paths below'''
    path_to_experiment = Path("experimentData.yml")
    path_to_change = Path("ChangeExperiment.py")
    path_to_mininet_run = Path("main.py")
    path_to_agent = Path("../ngtcp2-Science/")
    path_to_statlog_folder_in_agent = Path("perf/minimize_virt1_cwnd_p6/") # Path, where we trash all experiment logs
    path_to_minimization = path_to_agent / path_to_statlog_folder_in_agent / f"{curr_time_to_path()}.minimize" # The path to minimization log, mostly for testing
    path_to_optimal_params = path_to_agent / path_to_statlog_folder_in_agent / f"perfresCWND{curr_time_to_path()}.optimal"

    '''Ensure, that folder to save experiments exists'''
    if not os.path.exists(path_to_agent / path_to_statlog_folder_in_agent):
        os.makedirs(path_to_agent / path_to_statlog_folder_in_agent)

    '''CONFIG: Change parameters grid below'''
    segment_rtts = [10, 100]
    segment_bws = [40, 230]
    # selection_losses = [0.002, 0.005, 0.075, 0.5, 0.00125, 0.0015, 0.01, 0.015, 0.02, 0.1,
    #    0.2, 0.4, 0.05, 0.15, 0.0025, 0.0075, 0.025, 0.03, 0.04, 0.003,
    #    0.004, 0.0125, 0.3, 0.001, 0.75, 0.25, 3, 4, 5, 1.5, 2, 1, 1.25,
    #    2.5]
    selection_losses = [0.002, 0.005, 0.075, 0.5, 0.00125, 0.0015, 0.01, 0.015, 0.02, 0.1, 0.2, 0.4, 0.05, 0.15, 0.0025, 0.0075, 0.025, 0.03, 0.04, 0.003, 0.004, 0.0125, 0.3, 0.001, 0.75, 0.25, 3, 1.5, 2, 1, 1.25, 2.5]
    selection_jitts = [1] # Set jitter to 1ms, because w3 is not patched

    max_time_per_run = 360 # In minutes.
    expC = 24 * 7 * 4 # Manual calculation of experiment upper bound
    timeConstraint = 60 # If something goes wrong during experiment
    with open(path_to_minimization, 'a+') as minf:
        minf.write('Maximize speed for CWND parameter. ' + '\n\n')

    def optimize_cwnd(func, chan_params, x0, xmin, xmax, xeps, maxfev):
        cur_rtt, cur_loss, cur_bw, cur_jitt = chan_params
        fevi = 0
        xcur = x0
        while fevi < maxfev and xmax - xcur > xeps:
            fevi += 1
            speed, loss, rtt = func(xcur)
            sla_okay = (loss < cur_loss * 1.13 + 0.035) and (rtt < cur_rtt * 1.09)
            if sla_okay:
                xmin = xcur
            else:
                xmax = xcur
            xcur = int((xmin + xmax) / 2)

        optimize_dict = dict()
        optimize_dict["optimal_speed"] = speed
        optimize_dict["optimal_cwnd"] = xcur
        optimize_dict["nfev"] = fevi
        optimize_dict["xeps"] = xeps

        return optimize_dict

    def make_func_in_dot_real(cur_rtt, cur_loss, cur_bw, cur_jitt):
        def launch_experiment(cwnd):
            global expN, timeConstraint
            global path_to_statlog_folder_in_agent, path_to_agent, path_to_mininet_run, path_to_change, path_to_experiment
            answ_speeds = []
            answ_losss = []
            answ_rtts = []

            '''Experimentally change number of runs depending on speed variance'''
            launches_per_parameter_set = 5
            if cur_loss >= 0.1:
                launches_per_parameter_set = 6
            if cur_loss >= 1:
                launches_per_parameter_set = 7

            signal.signal(signal.SIGALRM, signal_handler)
            expN_aim = expN + launches_per_parameter_set
            while expN < expN_aim:
                expN += 1
                signal.alarm(timeConstraint)
                try:
                    t_one_start = time.time()
                    print(f"#### MinimizeBBRParamTest: Running experiment ({expN}/...) - rtt: {cur_rtt} loss: {cur_loss} bw: {cur_bw} jitt: {cur_jitt}\t... ")
                    print(f"#### CWND: {cwnd}\t ")

                    '''Assosiate statfile name with current second and number (optional)
                    (Assume, that no more than one exp during a second is held)'''
                    statfile = path_to_statlog_folder_in_agent / f"{expN}_exp_{curr_time_to_path()}"

                    '''Change some of the experiment parameters, others will stay constant'''
                    change_experiment(exp_path=str(path_to_experiment), p_rtt=cur_rtt, p_loss=cur_loss, p_bw=cur_bw, p_jitt=cur_jitt, cc='bbrfrcst', r_path=str(statfile), cong_window=cwnd)

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
                    if experiment_duration < 20:
                        expN -= 1

                        print(f"#### Experiment is too short: {experiment_duration} seconds")
                        with open(path_to_minimization, 'a+') as minf:
                            minf.write(f'Progress... Experiment is too short: {experiment_duration} seconds. skipping... \t' + '\n')
                    else:
                        answ_speed = datalist[-1]['bytes sent: '] / (float(datalist[0]['time'].split('-')[1]) - float(datalist[0]['time'].split('-')[0]))
                        answ_loss = datalist[-1]['mean loss2: '] * 100 # Fraction -> Percents
                        answ_rtt = datalist[-1]['mean s_rtt: ']
                        answ_speeds.append(answ_speed)
                        answ_losss.append(answ_loss)
                        answ_rtts.append(answ_rtt)

                        print(f"#### Current minimization answer: [speed: {answ_speed}, loss: {answ_loss}, rtt: {answ_rtt}].")
                        with open(path_to_minimization, 'a+') as minf:
                            minf.write(f'Progress... Current CWND: [{cwnd}]\t[speed: {answ_speed}, loss: {answ_loss}, rtt: {answ_rtt}] \t' + '\n')

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

            mean_answ_speed = (sum(answ_speeds) / len(answ_speeds))
            mean_answ_loss = (sum(answ_losss) / len(answ_losss))
            mean_answ_rtt = (sum(answ_rtts) / len(answ_rtts))
            with open(path_to_minimization, 'a+') as minf:
                minf.write(f"Returning answer: [speed: {mean_answ_speed}, loss: {mean_answ_loss}, rtt: {mean_answ_rtt}] in dot [{cur_rtt}, {cur_loss}, {cur_bw}, {cur_jitt}]\n")

            '''Return speed, loss, rtt to optimizer'''
            return mean_answ_speed, mean_answ_loss, mean_answ_rtt
        return launch_experiment

    '''Minimisation parameters'''
    cc = 'bbrfrcst'

    # The order is: 
    minimal_value = 50000
    maximal_value = 3500000

    start_value = 1250000

    minimization_method = '"Binary Search by Uberariy"'
    with open(path_to_optimal_params, 'a+') as resultf:
        yaml.dump([], resultf, default_flow_style=True) # We must create file with a+	        

    '''Run experiments'''
    t_start = time.time()
    for expN in range(expC):
        cur_rtt = int(np.random.uniform(segment_rtts[0], segment_rtts[1], 1)[0])
        cur_loss = random.choice(selection_losses)
        cur_bw = int(np.random.uniform(segment_bws[0], segment_bws[1], 1)[0])
        cur_jitt = random.choice(selection_jitts)
        func = make_func_in_dot_real(cur_rtt=cur_rtt, cur_loss=cur_loss, cur_bw=cur_bw, cur_jitt=cur_jitt)
        with open(path_to_minimization, 'a+') as minf:
            minf.write("Starting mimimzation. Method: " + minimization_method + "\n")
        res = optimize_cwnd(func, chan_params=[cur_rtt, cur_loss, cur_bw, cur_jitt], x0=start_value, xmin=minimal_value, xmax=maximal_value, xeps=300, maxfev=100)
        print(res["nfev"], "stand iterations\tfun: ", res["optimal_speed"], "\tcwnd: ", res["optimal_cwnd"])
        with open(path_to_minimization, 'a+') as minf:
            minf.write(f"!Finished in this dot. Time constraint: {max_time_per_run} minutes. Result:" + "\n" + str(res) + "\n\n")
        
        '''Save result'''
        info = dict()
        info['channel_rtt'] = cur_rtt
        info['channel_loss'] = cur_loss
        info['channel_bw'] = cur_bw
        info['channel_jitt'] = cur_jitt
        info['channel_congestion_control'] = cc
        info['res_nfev'] = float(res["nfev"])
        info['estimated_opt_speed'] = float(res["optimal_speed"])
        info['estimated_opt_cwnd'] = float(res["optimal_cwnd"])
        with open(path_to_optimal_params, 'r+') as resultf:
            datalist = yaml.safe_load(resultf) or []

        datalist.append(info)
        with open(path_to_optimal_params, 'w') as resultf:
            yaml.dump(datalist, resultf, default_flow_style=True)
                    
    t_end = time.time()

    print(f"#### Total time taken: {t_end - t_start} sec. ({(t_end - t_start) / 60} minutes)")
