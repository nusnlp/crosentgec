#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from collections import namedtuple
import numpy as np
import sys

import torch
from torch.autograd import Variable

from fairseq import data, options, tasks, tokenizer, utils
from fairseq.sequence_generator import SequenceGenerator
from fairseq.multiinput_sequence_generator import MultiInputSequenceGenerator


Batch = namedtuple('Batch', 'srcs tokens lengths ctxs ctx_tokens ctx_lengths')
Translation = namedtuple('Translation', 'src_str hypos alignments')


def buffered_read(buffer_size, input_files=None):
    buffer = []
    if input_files == None:
        for src_str in sys.stdin:
            buffer.append(src_str.strip())
            if len(buffer) >= buffer_size:
                yield buffer
                buffer = []
    else:
        for inps in zip(*[open(fi) for fi in input_files]):
            buffer.append([inp.strip() for inp in inps])
            if len(buffer) >= buffer_size:
                yield buffer
                buffer = []

    if len(buffer) > 0:
        yield buffer


def make_batches(inputs_buffer, args, src_dict, ctx_dict, max_positions ):
    ctx_tokens = [
    tokenizer.Tokenizer.tokenize(inputs[1], ctx_dict, add_if_not_exist=False).long()
    for inputs in inputs_buffer
    ]

    tokens = [
        tokenizer.Tokenizer.tokenize(inputs[0], src_dict, add_if_not_exist=False).long()
        for inputs in inputs_buffer
    ]

    src_sizes = np.array([t.numel() for t in tokens])
    ctx_sizes = np.array([t.numel() for t in ctx_tokens])
    #!debug
    if len(max_positions) < 3:
        max_positions += (max_positions[0],)
    itr = data.EpochBatchIterator(
        dataset=data.LanguageTripleDataset(
            src=tokens, src_sizes=src_sizes, src_dict=src_dict,
            ctx=ctx_tokens, ctx_sizes=ctx_sizes, ctx_dict=ctx_dict
            ),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=max_positions,
    ).next_epoch_itr(shuffle=False)

    for batch in itr:
        yield Batch(
            srcs=[inputs_buffer[i][0] for i in batch['id']],
            tokens=batch['net_input']['src_tokens'],
            lengths=batch['net_input']['src_lengths'],
            ctxs=[inputs_buffer[i][1] for i in batch['id']],
            ctx_tokens=batch['net_input']['ctx_tokens'],
            ctx_lengths=batch['net_input']['ctx_lengths']
        ), batch['id']


def main(args):
    if args.buffer_size < 1:
        args.buffer_size = 1
    if args.max_tokens is None and args.max_sentences is None:
        args.max_sentences = 1

    assert not args.sampling or args.nbest == args.beam, \
        '--sampling requires --nbest to be equal to --beam'
    assert not args.max_sentences or args.max_sentences <= args.buffer_size, \
        '--max-sentences/--batch-size cannot be larger than --buffer-size'

    print(args)

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Setup task, e.g., translation
    task = tasks.setup_task(args)

    # Load ensemble
    print('| loading model(s) from {}'.format(args.path))
    model_paths = args.path.split(':')
    models, model_args = utils.load_ensemble_for_inference(model_paths, task)

    # Set dictionaries
    src_dict = task.source_dictionary
    ctx_dict = task.context_dictionary
    tgt_dict = task.target_dictionary

    # Optimize ensemble for generation
    for model in models:
        model.make_generation_fast_(
            beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
        )

    # Initialize generator
    translator = MultiInputSequenceGenerator(
        models, tgt_dict, beam_size=args.beam, stop_early=(not args.no_early_stop),
        normalize_scores=(not args.unnormalized), len_penalty=args.lenpen,
        unk_penalty=args.unkpen, sampling=args.sampling, sampling_topk=args.sampling_topk,
        minlen=args.min_len,
    )

    if use_cuda:
        translator.cuda()

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(args.replace_unk)

    def make_result(src_str, hypos):
        result = Translation(
            src_str='O\t{}'.format(src_str),
            hypos=[],
            alignments=[],
        )

        # Process top predictions
        for hypo in hypos[:min(len(hypos), args.nbest)]:
            hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                hypo_tokens=hypo['tokens'].int().cpu(),
                src_str=src_str,
                alignment=hypo['alignment'].int().cpu(),
                align_dict=align_dict,
                tgt_dict=tgt_dict,
                remove_bpe=args.remove_bpe,
            )
            result.hypos.append('H\t{}\t{}'.format(hypo['score'], hypo_str))
            result.alignments.append('A\t{}'.format(' '.join(map(lambda x: str(utils.item(x)), alignment))))
        return result

    def process_batch(batch):
        tokens = batch.tokens
        lengths = batch.lengths
        ctx_tokens = batch.ctx_tokens
        ctx_lengths = batch.ctx_lengths

        if use_cuda:
            tokens = tokens.cuda()
            lengths = lengths.cuda()
            ctx_tokens = ctx_tokens.cuda()
            ctx_lengths = ctx_lengths.cuda()

        translations = translator.generate(
            src_tokens=Variable(tokens),
            src_lengths=Variable(lengths),
            ctx_tokens=Variable(ctx_tokens),
            ctx_lengths=Variable(ctx_lengths),
            maxlen=int(args.max_len_a * tokens.size(1) + args.max_len_b),
        )

        return [make_result(batch.srcs[i], t) for i, t in enumerate(translations)]

    if args.buffer_size > 1:
        print('| Sentence buffer size:', args.buffer_size)
    print('| Type the input sentence and press return:')
    for inputs_buffer in buffered_read(args.buffer_size, args.input_files):
        indices = []
        results = []
        for batch, batch_indices in make_batches(inputs_buffer, args, src_dict, ctx_dict, models[0].max_positions()):

            indices.extend(batch_indices)
            results += process_batch(batch)

        for i in np.argsort(indices):
            result = results[i]
            print(result.src_str)
            for hypo, align in zip(result.hypos, result.alignments):
                print(hypo)
                print(align)


if __name__ == '__main__':
    parser = options.get_generation_parser(interactive=True)
    args = options.parse_args_and_arch(parser)
    main(args)
