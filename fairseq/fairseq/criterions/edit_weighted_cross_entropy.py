# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
import torch
import torch.nn.functional as F
import sys

from fairseq import utils

from . import FairseqCriterion, register_criterion



@register_criterion('edit_weighted_cross_entropy')
class EditWeightedCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)

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

        target = model.get_targets(sample, net_output).view(-1)

        try:
            aligns = [self.alignments[idx] for idx in sample['id']]
            weighted_alignbatch = utils.prepare_align_batch(aligns, max_len=sample['target'].size(1), pad_idx=1)
            wlprobs = lprobs * weighted_alignbatch.view(-1,1)

        except:
            print('| error: alignment did not match, probably model is being validated (using standard cross entropy loss instead of edit weighted loss).')
            # weighted log probs is log probs itself (for validation and in case of error)
            wlprobs = lprobs

        loss = F.nll_loss(wlprobs, target, size_average=False, ignore_index=self.padding_idx,
                          reduce=reduce)
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'ntokens': sample['ntokens'],
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        agg_output = {
            'loss': loss_sum / sample_size / math.log(2),
            'sample_size': sample_size,
        }
        if sample_size != ntokens:
            agg_output['nll_loss'] = loss_sum / ntokens / math.log(2)
        return agg_output
