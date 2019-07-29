#!/usr/bin/python

import time
import numpy as np
import codecs
import argparse
import math
import os
import sys
import gzip
from collections import OrderedDict, defaultdict

# Initializing the logging module
import logging
import log_utils as L

# For feature functions
if sys.version_info[0] >= 3:
    import torch
    from torch.autograd import Variable
    if os.path.exists(os.path.dirname(os.path.realpath(__file__))+'/lib/pytorch_pretrained_bert'):
         sys.path.insert(0, os.path.dirname(os.path.realpath(__file__))+'/lib/pytorch_pretrained_bert/')
         from lib.pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

# For KenLM features
sys.path.insert(0, 'lib/kenlm_python/')
import kenlm


# For edit operations feature
from lib import levenshtein

logger = logging.getLogger(__name__)

ln10 = math.log(10)

import random
random.seed(1234)

def feature_extractor(features_line):
    """ extract features as a dictionary from the feature file """
    feature_score_dict = OrderedDict()
    pieces = features_line.split()
    feature_name = None
    feature_scores = []
    for piece in pieces:
        if piece[-1] == '=':
            if feature_name is not None:
                feature_score_dict[feature_name] = feature_scores
            feature_name = piece[:-1]
            feature_scores = []
        else:
            feature_scores.append(float(piece))
    feature_score_dict[feature_name] = feature_scores

    return(feature_score_dict)


class LM:
    def __init__(self, name, path, normalize=False, debpe=False):
        self.path = path
        c = kenlm.Config()
        c.load_method = kenlm.LoadMethod.LAZY
        self.model = kenlm.Model(path, c)
        self.name = name
        self.normalize = normalize
        self.debpe = debpe
        logger.info('Intialized ' + str(self.model.order) + "-gram language model: " + path)

    def get_name(self):
        return self.name

    def get_score(self, source, candidate, item_idx):
        if self.debpe:
            candidate = candidate.replace('@@ ','')
        lm_score = self.model.score(candidate)
        log_scaled = round(lm_score*ln10,4)
        if self.normalize == True:
            if len(candidate):
                return (log_scaled * 1.0 ) / len(candidate.split())
        return str(round(lm_score*ln10,4))

class SAMPLE:
    def __init__(self, name):
        self.name = name

    def get_score(self, source, candidate, item_idx):
        return str(0.5)

class WordPenalty:
    """
        Feature to caclulate word penalty, i.e. number of words in the hypothesis x -1
    """
    def __init__(self, name):
        self.name = name

    def get_score(self, source, candidate, item_idx):
        return str(-1 * len(candidate.split()))

class EditOps:
    """
        Feature to calculate edit operations, i.e. number of deletions, insertions and substitutions
    """
    def __init__(self, name, dels=True, ins=True, subs=True):
        self.name = name
        self.dels = ins
        self.ins = ins
        self.subs = subs

    def get_score(self, source, candidate, item_idx):
        src_tokens = source.split()
        trg_tokens = candidate.split()
        # Get levenshtein matrix
        lmatrix, bpointers = levenshtein.levenshtein_matrix(src_tokens, trg_tokens, 1, 1, 1)

        r_idx = len(lmatrix)-1
        c_idx = len(lmatrix[0])-1
        ld = lmatrix[r_idx][c_idx]
        d = 0
        i = 0
        s = 0
        bpointers_sorted = dict()

        for k, v in bpointers.items():
            bpointers_sorted[k] =sorted(v, key=lambda x: x[1][0])

        # Traverse the backpointer graph to get the edit ops counts
        while (r_idx != 0 or c_idx != 0):
            edit = bpointers_sorted[(r_idx,c_idx)][0]
            if edit[1][0] == 'sub':
                s = s+1
            elif edit[1][0] == 'ins':
                i = i+1
            elif edit[1][0] == 'del':
                d = d+1
            r_idx = edit[0][0]
            c_idx = edit[0][1]
        scores = ""
        if self.dels:
            scores += str(d) + " "
        if self.ins:
            scores += str(i) + " "
        if self.subs:
            scores += str(s) + " "
        return scores

class LexWeights:
    '''
    Use translation model from SMT p(w_f|w_e) using the alignment model from NMT
    '''
    def __init__(self, name, f2e=None, e2f=None, align_file=None, debpe=False):

        self.name = name
        if align_file:
            logger.info("Reading alignment file")
            self.align_dict = self.prepare_align_dict(align_file, debpe)
        self.f2e_dict = None
        if f2e:
            logger.info("Reading lex f2e file: " + f2e)
            self.f2e_dict =  self.prepare_lex_dict(f2e)
        self.e2f_dict = None
        if e2f:
            logger.info("Reading lex e2f file: " + e2f)
            self.e2f_dict = self.prepare_lex_dict(e2f)

        #for k in sorted(self.align_dict.iterkeys()):
            #print k, ":", self.align_dict[k].shape

    def set_align_file(align_file, debpe):
        logger.info("Reading alignment file")
        self.align_dict = self.prepare_align_dict(align_file, debpe)

    def prepare_lex_dict(self, lex_file):
        lex_dict = dict()
        with open(lex_file) as f:
            for line in f:
                pieces = line.strip().split()
                lex_dict[(pieces[0],pieces[1])] = math.log(float(pieces[2]))
        return lex_dict

    def prepare_align_dict(self, align_file, debpe):
        sent_count = -1
        item_count = 0
        align_dict = dict()
        aligns = []
        src_sent = ""
        candidate_sent = ""
        count = 1
        with open(align_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                pieces = line.split('|||')
                if len(pieces) > 1:

                    aligns = np.array(aligns)

                    ## Utility function to debpe aligns
                    def debpe_aligns(aligns, src_sent, candidate_sent):
                        src_tokens = src_sent.split()
                        candidate_tokens = candidate_sent.split()

                        # debug
                        src_debpe_tokens = src_sent.replace('@@ ','').split()
                        cand_debpe_tokens = candidate_sent.replace('@@ ','').split()

                        #print src_tokens, candidate_tokens, aligns.shape
                        assert aligns.shape == (len(candidate_tokens)+1, len(src_tokens)+1) or aligns.shape == (len(candidate_tokens), len(src_tokens)+1) , "Mismatch before debpe!" + str(aligns.shape) + " " + src_sent + " ( " + str(len(candidate_tokens)) + " ) " + " CAND:" + candidate_sent
                        before_shape = aligns.shape
                        ### Summing up and averaging across rows (candidate tokens) where BPE split occurs
                        start_idx = -1
                        end_idx = -1
                        delete_rows = []
                        for i in xrange(len(candidate_tokens)):
                            cand_token = candidate_tokens[i]
                            if len(cand_token)>=2 and cand_token[-2:] == '@@':
                                if start_idx == -1:
                                    start_idx = i
                                    end_idx = i
                                else:
                                    end_idx = i
                            else:
                                if start_idx != -1:
                                    aligns[start_idx] = np.sum(aligns[start_idx:end_idx+2], axis=0) / (end_idx - start_idx + 2)
                                    delete_rows += range(start_idx+1, end_idx+2)
                                    start_idx = -1

                        ### Summing up across columns (src_tokens) where BPE split occurs
                        start_idx = -1
                        end_idx = -1
                        delete_cols = []
                        for j in xrange(len(src_tokens)):
                            src_token = src_tokens[j]
                            if len(src_token) >= 2 and src_token[-2:]== '@@':
                                if start_idx == -1:
                                    start_idx = j
                                    end_idx = j
                                else:
                                    end_idx = j
                            else:
                                if start_idx != -1:
                                    aligns[:,start_idx] = np.sum(aligns[:, start_idx:end_idx+2], axis=1)
                                    delete_cols += range(start_idx+1, end_idx+2)
                                    start_idx = -1

                        #print aligns.shape, delete_rows, delete_cols
                        aligns = np.delete(aligns, delete_rows, axis=0)
                        aligns = np.delete(aligns, delete_cols, axis=1)

                        #print len(src_debpe_tokens), len(cand_debpe_tokens), aligns.shape, before_shape, src_tokens, src_debpe_tokens
                        #print src_tokens, len(src_tokens)
                        #print src_debpe_tokens, len(src_debpe_tokens)
                        #print candidate_tokens, len(candidate_tokens)
                        #print cand_debpe_tokens, len(cand_debpe_tokens)
                        #print before_shape, (len(candidate_tokens), len(src_tokens)), aligns.shape, (len(cand_debpe_tokens), len(src_debpe_tokens))
                        assert aligns.shape == (len(cand_debpe_tokens)+1, len(src_debpe_tokens)+1) or aligns.shape == (len(cand_debpe_tokens), len(src_debpe_tokens)+1), "mismatch after debpe!" + str(len(src_debpe_tokens))
                        return aligns

                    ### End of utility function ##

                    before_shape = aligns.shape
                    if sent_count>-1 and debpe == True:
                        aligns = debpe_aligns(aligns, src_sent, candidate_sent)
                    '''
                    if sent_count == 167 and item_count == 7:
                        #print aligns.shape
                        print "DEBUG"
                        src_debpe_tokens = src_sent.replace('@@ ','').split()
                        cand_debpe_tokens = candidate_sent.replace('@@ ','').split()
                        candidate_tokens = candidate_sent.split()
                        src_tokens = src_sent.split()
                        print src_debpe_tokens, len(src_debpe_tokens)
                        print candidate_tokens, len(candidate_tokens)
                        print cand_debpe_tokens, len(cand_debpe_tokens)
                        print before_shape, (len(candidate_tokens), len(src_tokens)), aligns.shape, (len(cand_debpe_tokens), len(src_debpe_tokens))
                    '''
                    align_dict[(sent_count, item_count)] = aligns
                    aligns = []
                    if int(pieces[0]) == sent_count:
                        item_count += 1
                    else:
                        assert sent_count + 1 == int(pieces[0]), "Malformed alignment file!"
                        sent_count =  sent_count+1
                        item_count = 0
                    src_sent = pieces[3]
                    candidate_sent = pieces[1]
                else:
                    weights = [float(piece) for piece in line.split()]
                    aligns.append(weights)

        aligns = np.array(aligns)
        if sent_count>-1 and debpe == True:
            aligns = debpe_aligns(aligns, src_sent, candidate_sent)
        align_dict[(sent_count, item_count)] = np.array(aligns)
        return align_dict

    def get_score(self, source, candidate, item_idx ):
        aligns = self.align_dict[item_idx]
        if (len(candidate.split())+1, len(source.split())+1) != aligns.shape and (len(candidate.split()), len(source.split())+1) != aligns.shape:
            print(source, candidate, aligns.shape, len(source.split()), len(candidate.split()))
        assert (len(candidate.split())+1, len(source.split())+1) == aligns.shape or (len(candidate.split()), len(source.split())+1) == aligns.shape, "Alignment dimension mismatch at: " + str(item_idx)
        candidate_tokens = candidate.split()
        source_tokens = source.split()
        f2e_score = 0.0
        e2f_score = 0.0
        for i in xrange(len(candidate_tokens)):
            for j in xrange(len(source_tokens)):
                #print "CANDIDATE_TOKEN:", candidate_tokens[i], "SOURCE_TOKEN:", source_tokens[j], "PROB:", self.f2e_dict[(candidate_tokens[i], source_tokens[j])], "ALIGN:", aligns[i,j]
                if self.f2e_dict:
                    if (candidate_tokens[i], source_tokens[j]) in self.f2e_dict:
                        f2e_score += self.f2e_dict[(candidate_tokens[i], source_tokens[j])]*aligns[i,j]
                    else:
                        f2e_score += math.log(0.0000001)*aligns[i,j]
                if self.e2f_dict:
                    if (source_tokens[j], candidate_tokens[i]) in self.e2f_dict:
                        e2f_score += self.e2f_dict[(source_tokens[j], candidate_tokens[i])]*aligns[i,j]
                    else:
                        e2f_score += math.log(0.0000001)*aligns[i,j]
        scores = ""
        if self.f2e_dict:
            scores += str(f2e_score) + " "
        if self.e2f_dict:
            scores += str(e2f_score) + " "

        return scores

class BERT:
    def __init__(self, name,cased=True, large=False):
        """
        Feature for BERT probs


        Args:
            name: feature name
            cased: (boolean) for lowercasing, set False
        """
        self.name = name
        self.cased = cased
        self.large = large
        self.MAXLEN = 150
        self.flogsoftmax = torch.nn.LogSoftmax(dim=1)

        model_name='bert-large' if large else 'bert-base'
        model_name = model_name+'-cased' if cased else model_name+'-uncased'
        print(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name,do_lower_case=not cased)
        self.model = BertForMaskedLM.from_pretrained(model_name).cuda()
        self.model.eval()

    def get_score(self, source, candidate, item_idx):
        if candidate == "":
            candidate = "."

        self.model.eval()
        with torch.no_grad():

            # add extra tokens [CLS] and [SEP]
            tokenized_text = ["[CLS]"] + self.tokenizer.tokenize(candidate)[0:self.MAXLEN] + ["[SEP]"]
            # converting to indices
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
            # find index of mask token
            mask_idx = self.tokenizer.convert_tokens_to_ids(['[MASK]'])[0]
            # converting to tensors
            tokens_tensor = torch.tensor([indexed_tokens]).cuda()
            segments_tensor = torch.zeros_like(tokens_tensor).cuda()
            tokens_tensor_exp = tokens_tensor.repeat(len(indexed_tokens)-2, 1)
            segments_tensor_exp = segments_tensor.repeat(len(indexed_tokens)-2, 1)
            # putting mask in tokens tensor
            idx_range = torch.arange(0,len(indexed_tokens)-2, out=torch.LongTensor())
            tokens_tensor_exp[idx_range,idx_range+1] = mask_idx
            # make predictions
            predictions = self.model(tokens_tensor_exp, segments_tensor_exp)[idx_range,idx_range+1]
            sum_log_probs = self.flogsoftmax(predictions)[idx_range,tokens_tensor[0,1:-1]].sum().item()
            # deleting variables to prevent out of memory
            del predictions
            del tokens_tensor_exp
            del segments_tensor_exp
            del segments_tensor
            del tokens_tensor

        return str(sum_log_probs)
