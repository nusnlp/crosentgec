# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
import torch
import torch.nn.functional as F

from fairseq import utils

from . import FairseqCriterion, register_criterion


@register_criterion('edit_weighted_label_smoothed_cross_entropy')
class EditWeightedLabelSmoothedCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing

        assert args.align_file is not None, "--align-file is required for edit weighted cross entropy loss"

        # prepare args
        self.edit_weight = args.edit_weight
        self.align_file = args.align_file

        # prepare alignments
        print('| preparing alignments for edit weighted cross entropy loss.')
        self.alignments = utils.get_weighted_alignments(self.align_file, self.edit_weight, task)
        print('| finished preparing aligment for {} examples.'.format(len(self.alignments)))


    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--edit-weight', '--ew', default=3.0, type=float, metavar='EW',
                       help='weight for edit weighted loss function.')
        parser.add_argument('--align-file', metavar='ALIGN', default=None,
                       help='an alignment file (optional: for edit weighted cross entropy loss)')


    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)
        non_pad_mask = target.ne(self.padding_idx)

        try:
            aligns = [self.alignments[idx] for idx in sample['id']]
            #  log(ew), exp(log(ew)*{0,1}) = 1 (for aligned) and ew (for unaligned)
            weighted_alignbatch = utils.prepare_align_batch(aligns, max_len=sample['target'].size(1), pad_idx=1)
            wlprobs = lprobs * weighted_alignbatch.view(-1,1)

        except:
            print('| error: alignment did not match, probably model is being validated (using label smoothed cross entropy loss without edit weighting).')
            # weighted log probs is log probs itself (for validation and in cae of error)
            wlprobs = lprobs


        # use weight log probs to find the NLL loss
        nll_loss = -wlprobs.gather(dim=-1, index=target)[non_pad_mask]

        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)[non_pad_mask]

        if reduce:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()
        eps_i = self.eps / lprobs.size(-1)
        loss = (1. - self.eps) * nll_loss + eps_i * smooth_loss

        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else loss.data,
            'ntokens': sample['ntokens'],
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        return {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size / math.log(2),
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / ntokens / math.log(2),
            'sample_size': sample_size,
        }
