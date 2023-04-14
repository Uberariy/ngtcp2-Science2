"""Main module to perfom experiments."""

from mininet.cli import CLI
from mininet.net import Mininet
from mininet.util import dumpNodeConnections
from mininet.log import setLogLevel

import NetworkSimulation as ns
import ExpDataGen
import BackgroundTrafficGen as btgen
import SdwanGen

import argparse
import sys
import yaml
from typing import Dict
from ExpParams import ExpParams
from pydantic import ValidationError


def parseParams( paramFile:str ) -> ExpParams:
    """Parse yaml file with experiment parameters into ExpParams object.

    params:
        paramFile - str, yaml file to parse
    returns:
        ExpParams object with parsed params
    """
    with open( paramFile, "r" ) as f:
        data = yaml.safe_load( f )
    data = ExpParams.parse_obj( data )
    return data


def runExperiment(args) -> None:
    """"Run experiments.
    
    params:
        args - argparse.ArgumentParser object with parsed params
    retuns:
        None 
    """
    # Generate experiments parameters, returns list with paths to experiments
    allExpParams = ExpDataGen.makeExperiments( expConf=args.expConfig, dir=args.expParams, randomState=0 )   # todo, mb clear 

    for i, expParams in enumerate( allExpParams ):
        # Parse experiment parameters
        try:
            tmpData  = parseParams( expParams )
        except ValidationError as e:
            print( f'exp{i}: Experiment parameters validation error:', e.json() )
            continue
        except Exception:
            print( f"exp{i}: Invalid experiment params" )
            continue
        topoParams, flowParams, backgroundParams = tmpData.topo, tmpData.flows, tmpData.background

        # Create and start network
        net = ns.makeNetwork( topoParams )
        net.start()
        print( "Dumping host connections" )
        dumpNodeConnections( net.hosts )
        
        # Background
        btgen.genBackgroundTraffic( net, backgroundParams )
        
        # SD-WAN
        SdwanGen.startSDWAN( net, flowParams, topoParams, logDir=args.logs )

        net.stop()

    return None


def createAndTest(args) -> None:
    """"Create topology and test it.
    
    params:
        args - argparse.ArgumentParser object with parsed params
    retuns:
        None 
    """
    # Parse experiment parameters
    try:
        tmpData  = parseParams( args.expConfig )
    except ValidationError as e:
        print( "Experiment parameters validation error:", e.json() )
        sys.exit()
    except Exception:
        print( "Invalid experiment params" )
        sys.exit()
    topoParams, flowParams, backgroundParams = tmpData.topo, tmpData.flows, tmpData.background

    # Create and start network
    net, bbrfrcstParams = ns.makeNetwork( topoParams )
    net.start()
    print( "Dumping host connections" )
    dumpNodeConnections(net.hosts)
    
    # Perform test
    if args.test:
        ns.iperfTest( net )
    
    # uberariy:
    print("ZXC", int(bbrfrcstParams.linkParams[0].rtt_mean.split('ms')[0]), 
				 float(bbrfrcstParams.linkParams[0].loss), 
				 int(bbrfrcstParams.linkParams[0].bw))
    p_rtt = int(bbrfrcstParams.linkParams[0].rtt_mean.split('ms')[0])
    p_loss = float(bbrfrcstParams.linkParams[0].loss)
    p_bw = int(bbrfrcstParams.linkParams[0].bw)
    logfile = "perf/perfres_2"
    
    h1, h2 = net.get( "h1", "h2" )
    h1.cmd("cd ../ngtcp2-Science")
    h2.cmd("cd ../ngtcp2-Science")
    h1.cmd(f"touch {logfile}")
    for iter_bw in [i*p_bw//10 for i in range(10, 11)]: # range(10, 11) equals to a normal experiment
        for i in range(3):
            with open("../ngtcp2-Science/"+logfile, "a") as f:
                f.write(f"\nBBRFRCSTexperiment. p_rtt: {p_rtt} p_loss: {p_loss} p_bw: {p_bw} bw givens to frcst in megabits: {iter_bw}\n")
            h1.cmd(f"examples/server --cc=bbrfrcst --bbrfrcst-params={p_rtt}ms,{p_loss},{iter_bw*128}K --no-quic-dump --no-http-dump 10.0.0.1 1234 ../keys/mycert-key.txt ../keys/mycert-cert.txt -d ../ 2> perf/logsrv{i} &")
            h2.cmd(f"examples/client --cc=bbr2 --no-quic-dump --no-http-dump 10.0.0.1 1234 https://10.0.0.1:1234/bigfile 2> perf/logcl{i}")
            h1.cmd("\c")
            h2.cmd("\c")
            h1.cmd(f"python3 perf/speedlog.py perf/logsrv{i} BYTE SENT 2500 --srtt=true --bw=true --loss=true --state=true >> {logfile}")	
    
    for i in range(2):
        with open("../ngtcp2-Science/"+logfile, "a") as f:
            f.write(f"\nBBR2experiment. p_rtt: {p_rtt} p_loss: {p_loss} p_bw: {p_bw}\n")
        h1.cmd(f"examples/server --cc=bbr2 --no-quic-dump --no-http-dump 10.0.0.1 1234 ../keys/mycert-key.txt ../keys/mycert-cert.txt -d ../ 2> perf/logsrv{i} &")
        h2.cmd(f"examples/client --cc=bbr2 --no-quic-dump --no-http-dump 10.0.0.1 1234 https://10.0.0.1:1234/bigfile 2> perf/logcl{i}")
        h1.cmd("\c")
        h2.cmd("\c")
        h1.cmd(f"python3 perf/speedlog.py perf/logsrv{i} BYTE SENT 2500 --srtt=true --bw=true --loss=true --state=true >> {logfile}")	
    
    # Enter CLI mode
    if args.cli:
        CLI( net ) 

    net.stop()
    return None


def parseArgs():
    parser = argparse.ArgumentParser(description="Runs experiment",
                                     usage="sudo ${PATH_TO_PYTHON_WITH_MININET_PACKAGE} "
                                         "%(prog)s expConfig.yml")
    parser.add_argument("expConfig", help="yaml file with experiment configuration")
    parser.add_argument("-d", "--expParamsDir", default='.', help="Directory to store experiment's parameters")
    parser.add_argument("-l", "--logs", default='.', help="Directory to store experiment's logs")
    parser.add_argument("--cli", action="store_true", 
                        help="enter CLI mode after iperf test ('exit' to leave)")
    parser.add_argument("--test", action="store_true", 
                        help="test topology links with iperf after network creation")                    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # Tell mininet to print useful information
    setLogLevel('info')
    args = parseArgs()
    createAndTest(args)
    #runExperiment(args)
    
