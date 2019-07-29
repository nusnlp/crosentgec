#!/usr/bin/env python

import sys
import os
import imp
import shutil

import argparse

# Initializing the logging module
import logging
import log_utils as L
import configreader
logger = logging.getLogger(__name__)



parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input-nbest", dest="input_nbest", required=True, help="Input n-best file")
parser.add_argument("-w", "--weights", dest="weights", required=True, help="Input weights file")
parser.add_argument("-o", "--output-dir", dest="out_dir", required=True, help="Output directory")
parser.add_argument("-c", "--clean-up", dest="clean_up", action='store_true', help="Temporary files will be removed")
parser.add_argument("-q", "--quiet", dest="quiet", action='store_true', help="Nothing will be printed in STDERR")
args = parser.parse_args()


from candidatesreader import NBestList
import codecs
import numpy as np

if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)
L.set_logger(os.path.abspath(args.out_dir),'train_log.txt')
L.print_args(args)


output_nbest_path = args.out_dir + '/augmented.nbest'
shutil.copy(args.input_nbest, output_nbest_path)

with open(args.weights, 'r') as input_weights:
    lines = input_weights.readlines()
    if len(lines) > 1:
        L.warning("Weights file has more than one line. I'll read the 1st and ignore the rest.")
    weights = np.asarray(lines[0].strip().split(" "), dtype=float)

prefix = os.path.basename(args.input_nbest)
input_aug_nbest = NBestList(output_nbest_path, mode='r')
output_nbest = NBestList(args.out_dir + '/' + prefix + '.reranked.nbest', mode='w')
output_1best = codecs.open(args.out_dir + '/' + prefix + '.reranked.1best', mode='w', encoding='UTF-8')

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

counter = 0
for group in input_aug_nbest:
    index = 0
    scores = dict()
    for item in group:
        features = np.asarray([x for x in item.features.split() if is_number(x)], dtype=float)
        try:
            scores[index] = np.dot(features, weights)
        except ValueError:
            logger.error('Number of features in the nbest and the weights file are not the same')
        index += 1
    sorted_indices = sorted(scores, key=scores.get, reverse=True)
    for idx in sorted_indices:
        output_nbest.write(group[idx])
    output_1best.write(group[sorted_indices[0]].hyp + "\n")
    counter += 1
    if counter % 100 == 0:
        logger.info(L.b_yellow(str(counter)) + " groups processed")
        logger.info("%i groups processed" % (counter))
logger.info("Finished processing %i groups" % (counter))
logger.info(L.green('Reranking completed.'))
output_nbest.close()
output_1best.close()

if args.clean_up:
    os.remove(output_nbest_path)
