import re
import sys

if __name__ == "__main__":
    # {experimentData.yml},{p_rtt},{p_loss},{p_bw}
    with open(sys.argv[1], 'r') as f:
        text = f.read()
    text_new1 = re.sub('bw_max      : \d*', r'bw_max      : '+sys.argv[4], text, flags = re.M)
    text_new2 = re.sub('bw_mean     : \d*', r'bw_mean     : '+sys.argv[4], text_new1, flags = re.M)
    text_new3 = re.sub('loss        : (\d|\.)*', r'loss        : '+sys.argv[3], text_new2, flags = re.M)
    text_new4 = re.sub('rtt_mean    : \d*', r'rtt_mean    : '+sys.argv[2], text_new3, flags = re.M)
    with open(sys.argv[1], 'w') as f:
        f.write(text_new4)