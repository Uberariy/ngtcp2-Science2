import re
import sys
from collections import defaultdict
import argparse
import json
import yaml


def pckt_dict(path, parti, fnd):
    '''Extract pckts per second and put in dict'''
    with open(path, 'r') as f:
        text = f.read()
    '''This reg. expr works for less than 10K seconds experiments'''
    patt = re.compile(r"I00(.*)[^I00]*"+fnd)
    d = defaultdict(int)
    for i in patt.findall(text):
        time_period = int(i[:6]) // parti
        d[time_period] += 1
    return(d)


def byte_dict(path, parti, fnd):
    '''Extract bytes per second and put in dict'''
    with open(path, 'r') as f:
        text = f.read()
    '''This reg. expr works for less than 10K seconds experiments'''
    patt = re.compile(r"I00(.*)[^I00]*"+fnd+"(.*)\n")
    d = defaultdict(int)
    for i in patt.findall(text):
        time_period = int(i[0][:6]) // parti
        d[time_period] += int(re.search(r" (\d*) bytes", i[1]).group(1))
    return(d)


def custom_dict(path, parti, fnd):
    '''Extract rtt/bw per second and put in dict'''
    with open(path, 'r') as f:
        text = f.read()
    '''This reg. expr works for less than 10K seconds experiments'''
    patt = re.compile(r"I00(.*)[^I00]*"+fnd+r"(.*)\n")
    if fnd.startswith("bbr2 "):
        d = defaultdict(str)
    else:
        d = defaultdict(int)
    cd = defaultdict(int)
    for i in patt.findall(text):
        time_period = int(i[0][:6]) // parti
        if fnd.startswith("loss"):
            buff = float(i[1].split()[0])
        elif fnd.startswith("bbr2 "):
            buff = str(i[1].split()[0]) + ","
        else:
            buff = int(i[1].split()[0])
        if fnd.startswith("min_rtt") and (buff >= 10**5):
            '''We encounter MAXINT, because no rtt is calculated'''
            d[time_period] += 0
        else:
            d[time_period] += buff
            cd[time_period] += 1
    if fnd.startswith("loss"):
        for i in d.keys():
            d[i] = d[i] / cd[i]
    elif not fnd.startswith("bbr2 "):
        for i in d.keys():
            d[i] = d[i] // cd[i]  
    return(d)


arg_parser = argparse.ArgumentParser(prog='speedlog',
                                     description='Calculate amount of client data sent. ' 
                                     'Note, that total packets calculated can be more, than content delivered, '
                                     'because we calculate content size of packets, that include content. ')                                     
arg_parser.add_argument('-v', '--version', action='version', version='%(prog)s 2.1')
arg_parser.add_argument('--lrtt', action='store_true', 
                        help='Use this parameter if you want mean latest rtt to be calculated. '
                             'Latest rtt is rtt from each packet. '
                             'It is better to work with srtt, because we dont include ack delays there')
arg_parser.add_argument('--srtt', action='store_true', 
                        help='Use this parameter if you want mean smoothed rtt to be calculated. '
                             'By default srtt is calculated in agent with 7/8 of previous value, and 1/8 of new '
                             'depending on pckts recieved')
arg_parser.add_argument('--mrtt', action='store_true', 
                        help='Use this parameter if you want mean min rtt to be calculated'
                             'Min rtt is congestion controller estimate on what min_rtt is')
arg_parser.add_argument('--bw', action='store_true', 
                        help='Use this parameter if you want mean bw to be calculated. '
                             'Max bw is congestion controller estimate on what max_bw is')
arg_parser.add_argument('--jitt', action='store_true', 
                        help='Use this parameter if you want mean jitter to be calculated')
arg_parser.add_argument('--loss', action='store_true', 
                        help='Use this parameter if you want mean loss percent to be calculated. ')
arg_parser.add_argument('--json', dest='json', type=str, default='', 
                        help='Save all the data in json format file with file path inserted (--json=PATH)')
arg_parser.add_argument('--yaml', dest='yaml', type=str, default='', 
                        help='Save all the data in json format file with file path inserted (--yaml=PATH)')

arg_parser.add_argument('--state', action='store_true',
                        help='Print states, that algo is in')

arg_parser.add_argument('path', metavar='PATH', nargs='?', type=str, default='',
                        help='Path of file with stderr of ngtcp2 client|server.')
arg_parser.add_argument('mode', metavar='MODE', nargs='?', type=str, default='',
                        choices=['BYTE', 'PCKT'], help='Specify mode of calculation. '
                        'Note, that these are UDP packets! ')
arg_parser.add_argument('direction', metavar='DIRECTION', nargs='?', type=str, default='',
                        choices=['SENT', 'RCVD'], help='Direction of packets calculated.')
arg_parser.add_argument('parti', metavar='PARTITION', nargs='?', type=str, default='',
                        help='Integer number, that provides time partition.')

if __name__ == '__main__':
    args = arg_parser.parse_args()
    filepath = args.path
    fnd = "Sent packet:" if args.direction.upper() == "SENT" else "Received packet:"
    try:
        parti = int(args.parti)
    except Exception as E:
        sys.exit(f"speedlog.py:\nPartition must be integer: {E}")
    print(f"Estimating speed in {args.mode.upper()} mode:")
    ld = []
    if args.mode.upper() == "BYTE":
        ld.append(byte_dict(filepath, parti, fnd))
    elif args.mode.upper() == "PCKT":
        ld.append(pckt_dict(filepath, parti, fnd))
    else:
        sys.exit("speedlog.py:\nNo such mode")
    whatis = [f'{args.mode.lower()}s {fnd.split()[0].lower()}: ']
    if args.srtt:
        ld.append(custom_dict(filepath, parti, "smoothed_rtt="))
        whatis.append("mean s_rtt: ")
    if args.lrtt:
        ld.append(custom_dict(filepath, parti, "latest_rtt="))
        whatis.append("mean l_rtt: ")
    if args.mrtt:
        ld.append(custom_dict(filepath, parti, "min_rtt="))
        whatis.append("mean min_rtt: ")
    if args.bw:
        ld.append(custom_dict(filepath, parti, "max_bw="))
        whatis.append("mean max_bw: ")
    if args.jitt:
        ld.append(custom_dict(filepath, parti, "rttvar="))
        whatis.append("mean jitter: ")
    if args.loss:
        ld.append(custom_dict(filepath, parti, "loss2="))
        whatis.append("mean loss: ")
    slaokay = True
    if args.state:
        tmpd1 = custom_dict(filepath, parti, "bbr2 enter ")
        tmpd2 = custom_dict(filepath, parti, "bbr2 start ")
        # print(tmpd1, tmpd2)
        for i, j in tmpd1.items():
            tmpd2[i] += j    
        for i in range(1, len(ld[0])):
            if i not in tmpd2.keys():
                tmpd2[i] = "== || ==" # tmpd2[i-1].split(',')[-2]+','
            elif tmpd2[i] != "Forecast,":
                slaokay = False
        ld.append(tmpd2)
        # print(tmpd1, tmpd2)
        whatis.append("states: ")
    savelist = []
    for i in ld[0].keys():
        pri = f"Second {i*parti/1000}-{(i+1)*parti/1000}:"
        print(pri, end=" "*(20 - len(pri)))
        pri = f"{whatis[0]}{ld[0][i]}"
        print(pri, end=" "*(24 - len(pri)))
        currd = dict()
        if args.json != '' or args.yaml != '':
            currd["time"] = f"{i*parti/1000}-{(i+1)*parti/1000}"
            currd[whatis[0]] = ld[0][i]
        for g, j in enumerate(ld[1:]):
            pri = f"{whatis[g+1]}{j[i]}"
            print(pri, end=" "*(24 - len(pri)))
            if args.json != '' or args.yaml != '':
                currd[whatis[g+1]] = j[i]
        if args.json != '' or args.yaml != '':
            savelist.append(currd)
        print("")
    if ld[0]:
        if len(ld[0].values()) != 1:
            '''We count mean speed ignoring last sample, because we are not sure if last period of time lasted completel'''
            print(f"Mean speed {sum([*ld[0].values()][:-1])/len([*ld[0].values()][:-1])} {args.mode.lower()}s per {parti/1000} seconds")
        else:
            print(f"Mean speed {[*ld[0].values()][0]} {args.mode.lower()}s per {parti/1000} seconds")
        if args.state:
            print("If it is bbrfrcst: ", end="")
            if slaokay:
                print("SLA IS OKAY")
            else:
                print("BAD SLA!")
    if args.json != '':
        with open(args.json, 'w') as f:
            json.dump(savelist, f, indent = 6)
    if args.yaml != '':
        with open(args.yaml, 'w') as f:
            yaml.dump(savelist, f, default_flow_style=True)
    sys.exit("Success")