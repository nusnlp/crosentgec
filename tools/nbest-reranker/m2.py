import sys
from features import feature_extractor
from candidatesreader import NBestList

from lib.m2scorer.scorer import levenshtein as m2_levenshtein
from lib.m2scorer.scorer.reader import load_annotation as m2_load_annotation

# Initializing the logging module
import logging
import log_utils as L
logger = logging.getLogger(__name__)

def m2_extractor(nbest_path, m2_ref, stats_file, features_file):
    nbest = NBestList(nbest_path, mode='r')
    group_count=-1
    source_sentences, gold_edits = m2_load_annotation(m2_ref, 'all')


    # M2Scorer Parameters
    max_unchanged_words=2
    beta = 0.5
    ignore_whitespace_casing= False
    verbose = False
    very_verbose = False

    with open(features_file, 'w') as ffeat, open(stats_file, 'w') as fstats:
        for group, source_sentence, golds_set in zip(nbest,source_sentences, gold_edits):
            group_count += 1
            candidate_count=-1
            candidates =  list(group)
            for candidate in candidates:
                candidate_count += 1
                feature_score_dict = feature_extractor(candidate.features)
                # write to features file
                p,r,f, stats = m2_levenshtein.batch_multi_pre_rec_f1([candidate.hyp], [source_sentence], [golds_set],  max_unchanged_words, beta, ignore_whitespace_casing, verbose, very_verbose, stats=True )
                if candidate_count == 0:
                    # header for each group
                    feature_list = ['{}_{}'.format(feature_name, idx) for feature_name, feature_values in feature_score_dict.items()  for idx, feature_value in enumerate(feature_values)  ]
                    ffeat.write("FEATURES_TXT_BEGIN_0 {} {} {} {}\n".format(group_count, len(candidates), len(feature_list), ' '.join(feature_list)))
                    num_stats = 4
                    fstats.write("SCORES_TXT_BEGIN_0 {} {} {} M2Scorer\n".format(group_count, len(candidates), num_stats))
                # write each line to features/stats
                ffeat.write(' '.join([' '.join([str(val) for val in feature_values]) for feature_values in feature_score_dict.values()]) + '\n')
                fstats.write('{} {} {} {} \n'.format(stats['num_correct'], stats['num_proposed'], stats['num_gold'], stats['num_src_tokens']))
            # footer for each group
            ffeat.write('FEATURES_TXT_END_0\n')
            fstats.write('SCORES_TXT_END_0\n')
            # logging
            logger.info("processed {} groups".format(group_count))
