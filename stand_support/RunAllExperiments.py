import sys
import os
import time

if __name__ == "__main__":
    # Change file pathes below
    path_to_experiment = "experimentData.yml"
    path_to_change = "ChangeExperminet.py"
    path_to_mininet_run = "main.py"
    # Change parameters grid below
    rtts = [40, 80]
    bws = [80, 160]
    losses = [0.1]
    t_start = time.time()
    expC = len(rtts)*len(bws)*len(losses)
    expN = 0
    for i in rtts:
        for j in losses:
            for g in bws:
				expN += 1
                t_one_start = time.time()
                print(f"#### RunAll: Running experiment ({expN}/{expC}) - rtt: {i} loss: {j} bw: {g}\t... ")
                os.system(f"python3 {path_to_change} {path_to_experiment} {i} {j} {g}")
                os.system(f"sudo /usr/bin/python3 {path_to_mininet_run} {path_to_experiment}")
                t_one_end = time.time()
                print(f"#### RunAll: Experiment finished. Time taken: {t_one_end - t_one_start} ({(t_one_end - t_one_start) // 60} minutes)")
    t_end = time.time()
    print(f"RunAll: Total time taken: {t_end - t_start} ({(t_end - t_start) // 60} minutes)")
