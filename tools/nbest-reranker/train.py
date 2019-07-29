#!/usr/bin/env python

import sys
import os
import shutil
import imp

import argparse

# Initializing the logging module
import logging
import log_utils as L
import configreader
logger = logging.getLogger(__name__)


import m2

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input-nbest", dest="input_nbest", required=True, help="Input n-best file")
parser.add_argument("-r", "--reference-files", dest="ref_paths", required=True, help="A comma-seperated list of reference files")
parser.add_argument("-c", "--config", dest="input_config", required=True, help="Input config (ini) file, e.g similar to moses with [weight] section")
parser.add_argument("-o", "--output-dir", dest="out_dir", required=True, help="Output directory")
parser.add_argument("-t", "--threads", dest="threads", default = 14, type=int, help="Number of MERT threads")
parser.add_argument("--no-add-weight", dest="no_add_weight", action="store_true", help="Flag to be true if config file already contains initial weights for augmented feature(s). Useful for adding multiple features.")
parser.add_argument("-iv", "--init-value", dest="init_value", default = '0.05', help="The initial value of the feature")
parser.add_argument("-a", "--tuning-algorithm", dest="alg", default = 'mert', help="Tuning Algorithm (mert|pro|wpro)")
parser.add_argument("-m", "--tuning-metric", dest="metric", default = 'bleu', help="Tuning Algorithm (bleu|m2)")
parser.add_argument("-s", "--predictable-seed", dest="pred_seed", action='store_true', help="Tune with predictable seed to avoid randomness")
parser.add_argument("--moses-dir", dest="moses_dir", required=True, help="Path to Moses. Required for tuning scripts")
args = parser.parse_args()

fscore_arg = ""
if args.metric == 'm2':
    fscore_arg = " --sctype M2SCORER --scconfig ignore_whitespace_casing:true "
    logger.info("Using M2 Tuning")
    logger.info(L.b_yellow('Arguments: ') + fscore_arg)


if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)

L.set_logger(os.path.abspath(args.out_dir),'train_log.txt')
L.print_args(args)

logger.info("Reading weights from config file")
features = configreader.parse_ini(args.input_config)
logger.info("Feature weights: " + str(features))

output_nbest_path = args.out_dir + '/augmented.nbest'
shutil.copy(args.input_nbest, output_nbest_path)

logger.info('Extracting stats and features')
logger.warning('The optional arguments of extractor are not used yet')

if args.metric == 'bleu':
    cmd = args.moses_dir + '/bin/extractor -r ' + args.ref_paths + ' -n ' + output_nbest_path + ' --scfile ' + args.out_dir + '/statscore.data --ffile ' + args.out_dir + '/features.data'
    logger.info('Executing command: ' + cmd )
    os.system(cmd)
if args.metric == 'm2':
    m2.m2_extractor(nbest_path=output_nbest_path, m2_ref=args.ref_paths, stats_file=args.out_dir + '/statscore.data', features_file=args.out_dir + '/features.data')
    #cmd = args.moses_dir + '/bin/extractor --sctype M2SCORER --scconfig ignore_whitespace_casing:true -r ' + args.ref_paths + ' -n ' + output_nbest_path + ' --scfile ' + args.out_dir + '/statscore.data --ffile ' + args.out_dir + '/features.data'

#create the list of features

with open(args.out_dir + '/init.opt', 'w') as init_opt:
    init_list = []
    for line in features:
        tokens = line.split(" ")
        try:
            float(tokens[1])
            init_list += tokens[1:]
        except ValueError:
            pass
    if args.no_add_weight == False:
        init_list.append(args.init_value)
    dim = len(init_list)
    init_opt.write(' '.join(init_list) + '\n')
    init_opt.write(' '.join(['0' for i in range(dim)]) + '\n')
    init_opt.write(' '.join(['1' for i in range(dim)]) + '\n')

seed_arg = ''
if args.pred_seed:
    seed_arg = ' -r 1 '
    #seed_arg = ' -r 1500 '


if (args.alg == 'mert'):
    logger.info('Running MERT')
    cmd = args.moses_dir + '/bin/mert -d ' + str(dim) + ' -S ' + args.out_dir + '/statscore.data -F ' + args.out_dir + '/features.data --ifile ' + args.out_dir + '/init.opt --threads ' + str(args.threads) + seed_arg + fscore_arg# + "-m 50 -n 20"
    logger.info("Command: " +  cmd)
    os.system(cmd)
else:
    logger.error('Invalid tuning algorithm: ' + args.alg)

logger.info(L.green("Optimization complete."))
assert os.path.isfile('weights.txt')
shutil.move('weights.txt', args.out_dir + '/weights.txt')

