#!/usr/bin/python

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input-file',  help='path to input file (output of fairseq)')
parser.add_argument('--debpe',  action='store_true', help='enable the flag to post-process and remove BPE segmentation.')
parser.add_argument('--retain-last-only',  action='store_true', help='retain the portion of the hypothesis after the last <CONCAT>')
args = parser.parse_args()


scount = -1
with open(args.input_file) as f:
    for line in f:
        line = line.strip()
        pieces = line.split('\t')
        if pieces[0] == 'O':
            scount += 1
        if pieces[0] == 'H':
            hyp = pieces[2]
            if args.debpe:
                hyp = hyp.replace('@@ ','')
            if args.retain_last_only:
                hyp_pieces = hyp.split('<CONCAT> ')
                hyp = hyp_pieces[-1]
            score = pieces[1]
            print("%d ||| %s ||| F0= %s ||| %s" % (scount, hyp, score, score) )


