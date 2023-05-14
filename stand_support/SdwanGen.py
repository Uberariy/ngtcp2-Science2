"""Application for sdwan  flows manipulation."""

import os
import sched
import threading
from mininet.net import Mininet
from ExpParams import AgentParams, BBR, SdwanFlow, TopoParams
from pathlib import Path
from typing import Dict
from calculate_channel_speed import load_and_predict

os.umask(0)

def opener(path, flags):
    return os.open(path, flags, 0o777)

def runFlow( net: Mininet, flowParams: SdwanFlow, topoParams: TopoParams, bbrParams: BBR, 
             agentParams: AgentParams, logDir: Path ) -> None:
    """Generate SDWAN flows and start transport agent."""
    # Link's number corresponds to network.
    # Get parameters to use in statistics
    # Right now, we are working with a single (first) link.
    p_rtt = int(topoParams.linkParams[0].rtt_mean.split('ms')[0])
    p_loss = float(topoParams.linkParams[0].loss)
    p_bw = int(topoParams.linkParams[0].bw)
    p_jt = float(topoParams.linkParams[0].jitter_max.split('ms')[0])

    cong_wind_bbrfrcst = int(topoParams.linkParams[0].cong_window)
    sla_r_speed = load_and_predict(path_to_models='speed_models/', cc='bbrfrcst', rtt=p_rtt, loss=p_loss, bw=1024*p_bw) # Convert to Kbit/s
    sla_r_speed = int(min(sla_r_speed, p_bw*1024))

    h1, h2 = net.get( "h1", "h2" )
    h1.cmd(f"cd {agentParams.agent_path}")
    h2.cmd(f"cd {agentParams.agent_path}")
    
    # Start agent
    # with open(os.path.join(str(agentParams.agent_path), str(agentParams.result_path)), "a", opener=opener) as f:
        # This syntax is important for statistics aggregation
        # f.write(f"\n{agentParams.cong_control.upper()}experiment. p_rtt: {p_rtt} p_loss: {p_loss} p_bw: {p_bw} p_jt: {p_jt}\n")
    
    cc_params = ""
    states = ""
    duration = int(flowParams.timeStop - flowParams.timeStart)
    if agentParams.cong_control == "bbrfrcst":
        cc_params += f"--bbrfrcst-params={p_rtt},{p_loss},{p_bw*1024*1024//8},{sla_r_speed*1024//8} --cong_wind_bbrfrcst={cong_wind_bbrfrcst}"
        states += "--states"
    # FOR DIPLOMA:
    # os.environ["LD_LIBRARY_PATH"] = "/home/admsys/inopsy/ngtcp2-Science/lib/.libs/:/home/admsys/inopsy/nghttp3/lib/.libs/:/home/admsys/inopsy/openssl/:/home/admsys/inopsy/ngtcp2-Science/crypto/openssl/.libs/"
    # os.environ['LD_LIBRARY_PATH'] = os.getcwd()
    h1.cmd("export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/admsys/inopsy/ngtcp2-Science/lib/.libs/:/home/admsys/inopsy/nghttp3/lib/.libs/:/home/admsys/inopsy/openssl/:/home/admsys/inopsy/ngtcp2-Science/crypto/openssl/.libs/")
    h2.cmd("export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/admsys/inopsy/ngtcp2-Science/lib/.libs/:/home/admsys/inopsy/nghttp3/lib/.libs/:/home/admsys/inopsy/openssl/:/home/admsys/inopsy/ngtcp2-Science/crypto/openssl/.libs/")
    h1.cmd(f"examples/server --cc={agentParams.cong_control} --no-quic-dump --no-http-dump --quit-timeout={agentParams.exp_max_dur}s " + \
           f"{agentParams.server_addr} {agentParams.server_port} {agentParams.key_path} {agentParams.cert_path} " + \
           f"--inopsy-log {cc_params} --filesize-zero=9G -d ../ 2> perf/logsrv &")
    h2.cmd(f"examples/client --cc=bbr2 --no-quic-dump --no-http-dump --quiet {agentParams.server_addr} " + \
           f"{agentParams.server_port} https://{agentParams.server_addr}:{agentParams.server_port}/{agentParams.file_path} --inopsy-log 2> perf/logcl")
    # FOR WORK:
    #h1.cmd(f"examples/server --cc=bbr2 --no-quic-dump --no-http-dump --quit-timeout={agentParams.exp_max_dur}s {cc_params} " + \
    #       f"{agentParams.server_addr} {agentParams.server_port} {agentParams.key_path} {agentParams.cert_path} " + \
    #       f"--inopsy-log --filesize-zero=15G -d ../ 2> perf/logsrv &")
    #h2.cmd(f"examples/client --cc={agentParams.cong_control} --no-quic-dump --no-http-dump --quiet {agentParams.server_addr} " + \
    #       f"{agentParams.server_port} https://{agentParams.server_addr}:{agentParams.server_port}/{agentParams.file_path} --inopsy-log 2> perf/logcl")
    #Statlog.py is called with standard parameters, that we want to analyse
    #FOR WORK OR OPTIMIZATION: In ngtcp2-Science gridsearch we call it in script
    os.system(f"python3 {agentParams.agent_path}/perf/statlog.py {agentParams.agent_path}/perf/logsrv BYTE SENT 1000 " + \
              f"--srtt --cwnd --loss --jitt --loss2 {states} --yaml={agentParams.agent_path}/{agentParams.result_path}")

    return


def startSDWAN(net: Mininet, flowParams: Dict[str, SdwanFlow], topoParams: TopoParams,
               bbrParams: BBR, agentParams, logDir: Path) -> threading.Thread:
    """Make a schedule and run all sdwan flowss.

    returns:
        threading.Thread - thread that generates flows
    """
    schedule = sched.scheduler()

    if len(flowParams) > 1:    # Remove in further stages
        print("Warning: at the moment only 1 sdwan flow generation is supported. "
              "Only first flow will be generated.")

    # Schedule flows
    for name, flow in flowParams.items():
        schedule.enter( flow.timeStart, 1, runFlow, argument=(net, flow, topoParams, bbrParams, agentParams),
                        kwargs={'logDir': logDir} )
        break   # run exactly one flow (to remove)

    # Schedule empty event when all flows ends
    workTime = max( flowParams.values(), key=lambda x: x.timeStop ).timeStop
    schedule.enter( workTime, 0, lambda *args: None )

    sdwanThread = threading.Thread( target=schedule.run )
    sdwanThread.start()

    return sdwanThread
