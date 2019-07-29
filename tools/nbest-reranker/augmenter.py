#!/usr/bin/python

import os
import time
import numpy as np
import codecs
import argparse
import datetime

# Initializing the logging module
import logging
import log_utils as L
logger = logging.getLogger(__name__)

from candidatesreader import NBestList
from features import *

def augment(features, source_path, input_nbest_path, output_nbest_path):
    ''' Function to augment the n-best list with a feature function
     :param feature: The feature function object
     :param source_path: Path to the original source sentences (maybe required for the feature function)
     :param input_nbest_path: Path to the n-best file
     :param output_nbest_path: Path to the output n-best file
    '''
    # Initialize NBestList objects
    logger.info('[{}] initializing n-best list'.format(datetime.datetime.now()))
    input_nbest = NBestList(input_nbest_path, mode='r')
    output_nbest = NBestList(output_nbest_path, mode='w')

    # Load the source sentences
    logger.info('[{}] loading source sentences'.format(datetime.datetime.now()))
    src_sents = codecs.open(source_path, mode='r', encoding='UTF-8')

    # For each of the item in the n-best list, append the feature
    sent_count = 0
    for group, src_sent in zip(input_nbest, src_sents):
        candidate_count = 0
        for item in group:
            for feature in features:
                item.append_feature(feature.name, feature.get_score(src_sent, item.hyp, (sent_count, candidate_count)))
            output_nbest.write(item)
            candidate_count += 1
        sent_count += 1
        if (sent_count % 100 == 0):
            logger.info('[{}] augmented '.format(datetime.datetime.now()) + L.b_yellow(str(sent_count)) + ' sentences.')
    output_nbest.close()


parser = argparse.ArgumentParser()
parser.add_argument("-s", "--source-sentence-file", dest="source_path", required=True, help="Path to the file containing source sentences.")
parser.add_argument("-i", "--input-nbest", dest="input_nbest_path", required=True, help="Input n-best file")
parser.add_argument("-o", "--output-nbest", dest="output_nbest_path", required=True, help="Output n-best file")
parser.add_argument("-f", "--feature", dest="feature_string", required=True, help="feature initializer, e.g. LM('LM0','/path/to/lm_file', normalize=True)")
args = parser.parse_args()

L.set_logger(os.path.abspath(os.path.dirname(args.output_nbest_path)),'augment_log.txt')
L.print_args(args)
features = eval('['+args.feature_string+']')
augment(features, args.source_path, args.input_nbest_path, args.output_nbest_path)
logger.info(L.green('Augmenting done.'))
