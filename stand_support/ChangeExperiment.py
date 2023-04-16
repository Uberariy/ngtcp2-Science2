import re
import sys
import argparse
import yaml
from collections import defaultdict

def change_experiment(exp_path, p_rtt=None, p_loss=None, p_bw=None, p_jitt=None, cc=None, r_path=None, bbr_beta=None, bbr_loss_tresh=None, bbr_probe_rtt_cwnd_gain=None, bbr_probe_rtt_duration=None, cong_window=None):
    '''Overwrites experimentData'''

    with open(exp_path, "r") as f:
        data = yaml.safe_load(f)
    
    data_first_link = data['Topo']['linkParams'][0]
    if p_bw is not None:
        data_first_link['bw_max'] = int(p_bw)
        data_first_link['bw_mean'] = int(p_bw)
    if p_loss is not None:
        data_first_link['loss'] = float(p_loss)
    if p_rtt is not None:
        data_first_link['rtt_mean'] = str(int(p_rtt)) + "ms"
    if p_jitt is not None:
        data_first_link['jitter_max'] = str(int(p_jitt)) + "ms"
    if cong_window is not None:
        data_first_link['cong_window'] = int(cong_window)
    data['Topo']['linkParams'][0] = data_first_link

    data_agent = data['transpAgentParams']
    if cc is not None:
        data_agent['cong_control'] = str(cc)
    if r_path is not None:
        data_agent['result_path'] = str(r_path)
    data['transpAgentParams'] = data_agent

    data_bbr = data['bbrGen']
    if bbr_beta is not None:
        data_bbr['BBRBeta'] = float(bbr_beta)
    if bbr_loss_tresh is not None:
        data_bbr['BBRLossTresh'] = float(bbr_loss_tresh)
    if bbr_probe_rtt_cwnd_gain is not None:
        data_bbr['BBRProbeRttCwndGain'] = float(bbr_probe_rtt_cwnd_gain)
    if bbr_probe_rtt_duration is not None:
        data_bbr['ProbeRTTDuration'] = int(bbr_probe_rtt_duration)
    data['bbrGen'] = data_bbr

    with open(exp_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, indent=4)

def change_experiment_regex(exp_path, p_rtt=None, p_loss=None, p_bw=None, p_jitt=None, cc=None, r_path=None, bbr_beta=None, bbr_loss_tresh=None, bbr_probe_rtt_cwnd_gain=None, bbr_probe_rtt_duration=None, cong_window=None):
    '''Saves comments in experimentData, but is less reliable'''
    def resub(arg1, arg2, check, text):
        if check is not None:
            return re.sub(arg1, arg2, text, flags = re.M)
        return text
    def str_(s):
        if s is not None:
            return str(s)
        return ''
    def int_(s):
        if s is not None:
            return int(s)
        return 0

    with open(exp_path, 'r') as f:
        text = f.read()
    text = resub('bw_max      : \d*', r'bw_max      : ' + str_(int_(p_bw)), p_bw, text)
    text = resub('bw_mean     : \d*', r'bw_mean     : ' + str_(int_(p_bw)), p_bw, text)
    text = resub('loss        : (\d|\.)*', r'loss        : ' + str_(p_loss), p_loss, text)
    text = resub('rtt_mean    : \d*', r'rtt_mean    : ' + str_(int_(p_rtt)), p_rtt, text)
    text = resub('jitter_max  : \d*', r'jitter_max  : ' + str_(int_(p_jitt)), p_jitt, text)
    text = resub('cong_control: [a-zA-Z0-9]+', r'cong_control: ' + str_(cc), cc, text)
    text = resub('result_path : .*', r'result_path : ' + str_(r_path), r_path, text)
    text = resub('BBRBeta            : .*', r'BBRBeta            : ' + str_(bbr_beta), bbr_beta, text)
    text = resub('BBRLossTresh       : .*', r'BBRLossTresh       : ' + str_(bbr_loss_tresh), bbr_loss_tresh, text)
    text = resub('BBRProbeRttCwndGain: .*', r'BBRProbeRttCwndGain: ' + str_(bbr_probe_rtt_cwnd_gain), bbr_probe_rtt_cwnd_gain, text)
    text = resub('ProbeRTTDuration   : .*', r'ProbeRTTDuration   : ' + str_(bbr_probe_rtt_duration), bbr_probe_rtt_duration, text)
    text = resub('cong_window : .*', r'cong_window : ' + str_(int_(cong_window)), cong_window, text)
    with open(exp_path, 'w') as f:
        f.write(text)

arg_parser = argparse.ArgumentParser(prog='ChangeExperiment',
                                     description='Change data for the next experiment in a stand with a regex. ')                                     
arg_parser.add_argument('-v', '--version', action='version', version='%(prog)s 2.0')

arg_parser.add_argument('--p_rtt', dest='p_rtt', type=float, default=40,  
                        help='Channel parameter')
arg_parser.add_argument('--p_bw', dest='p_bw', type=float, default=180,  
                        help='Channel parameter')
arg_parser.add_argument('--p_loss', dest='p_loss', type=float, default=0.001,  
                        help='Channel parameter')
arg_parser.add_argument('--p_jitt', dest='p_jitt', type=float, default=1,  
                        help='Channel parameter')
arg_parser.add_argument('--cc', dest='cc', type=str, default='bbrfrcst',  
                        help='Channel parameter')
arg_parser.add_argument('--bbr_beta', dest='bbr_beta', type=float, default=0.7,  
                        help='Channel parameter - used sometimes if cc is bbr2')
arg_parser.add_argument('--bbr_loss_tresh', dest='bbr_loss_tresh', type=float, default=0.02,  
                        help='Channel parameter - used sometimes if cc is bbr2')
arg_parser.add_argument('--bbr_probe_rtt_cwnd_gain', dest='bbr_probe_rtt_cwnd_gain', type=float, default=0.5,  
                        help='Channel parameter - used sometimes if cc is bbr2')
arg_parser.add_argument('--bbr_probe_rtt_duration', dest='bbr_probe_rtt_duration', type=float, default=200,  
                        help='Channel parameter - used sometimes if cc is bbr2')
arg_parser.add_argument('--speed', dest='speed', type=float, default=200,  
                        help='Channel parameter - used rarely')
arg_parser.add_argument('--cong_window', dest='cong_window', type=float, default=200,  
                        help='Channel parameter - not used in InOpSy')

arg_parser.add_argument('--r_path', dest='r_path', type=str, default='',  
                        help='Stand parameter')

arg_parser.add_argument('exp_path', metavar='EXP_PATH', nargs='?', type=str, default='',
                        help='Path to experimentData.yml')

if __name__ == "__main__":
    args = arg_parser.parse_args()
    change_experiment_regex(exp_path=args.exp_path, p_rtt=args.p_rtt, p_loss=args.p_loss, p_bw=args.p_bw, p_jitt=args.p_jitt, cc=args.cc, r_path=args.r_path, bbr_beta=args.bbr_beta, bbr_loss_tresh=args.bbr_loss_tresh, bbr_probe_rtt_cwnd_gain=args.bbr_probe_rtt_cwnd_gain, bbr_probe_rtt_duration=args.bbr_probe_rtt_duration, cong_window=args.cong_window)
    # change_experiment(exp_path=args.exp_path)