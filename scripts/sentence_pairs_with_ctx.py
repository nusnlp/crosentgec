# This script extracts sentence pairs from Lang-8, NUCLE and CoNLL-2014 data with XML format
# Python version: 2
# NLTK version: 2.0b7
# Usage example for Lang-8: python2 sentence_pairs_with_ctx.py --train --tokenize --maxtokens 80 --mintokens 1 --input lang8-train.xml  --src-ctx lang8.src-trg.ctx --src-src lang8.src-trg.src --trg-trg lang8.src-trg.trg
# Usage example for NUCLE: python2 sentence_pairs_with_ctx.py --train(--dev) --maxtokens 80 --mintokens 1 --input nucle-train(-dev).xml --src-ctx nucle(-dev).src-trg.ctx --src-src nucle(-dev).src-trg.src --trg-trg nucle(-dev).src-trg.trg
# Usage example for CoNLL-2014: python2 sentence_pairs_with_ctx.py --test --input conll14st-test.xml --src-ctx conll14st-test.tok.ctx --src-src conll14st-test.tok.src --trg-trg conll14st-test.tok.trg

import argparse
import xml.etree.cElementTree as ET
import nltk
parser = argparse.ArgumentParser()
parser.add_argument('--train', action='store_true', help='choose if we generate train data')
parser.add_argument('--dev', action='store_true', help='choose if we generate development data')
parser.add_argument('--test', action='store_true', help='choose if we generate test data')
parser.add_argument('--tokenize', action='store_true', help='choose if the dataset need to be tokenized.')
parser.add_argument('--maxtokens', type=int, help='set the maximum number of tokens in one sentence')
parser.add_argument('--mintokens', type=int, help='set the minimum number of tokens in one sentence')
parser.add_argument('--input', help='XML file need to be parsed')
parser.add_argument('--src-ctx', help='store context source sentences')
parser.add_argument('--src-src', help='store current source sentences')
parser.add_argument('--trg-trg', help='store current target sentences')
args = parser.parse_args()

def extract_from_xml(tokenize, input, src_ctx, src_src, trg_trg):
    count=0
    essay_n = 0
    source_n = 0
    token_n = 0
    token_src_ctx = 0
    token_src_src = 0
    token_trg_trg = 0
    with open(src_ctx,'w') as ctx_file:
        with open(src_src,'w') as src_file:
            with open(trg_trg,'w') as tag_file:
                tree=ET.parse(input)
                root=tree.getroot()
                for essay in root:
                    essay_n += 1
                    ctx_1 = ''
                    ctx_2 = ''
                    source_ctx = '\n'
                    cur_ctx_token =0
                    for sentence in essay:
                        source=sentence.find('source')
                        cache_src = []
                        if source.get('langid')!='en':
                            continue
                        else:
                            if tokenize:
                                tokens_src = nltk.word_tokenize(source.text.strip())
                                for each in tokens_src:
                                    cache_src.append(each)
                                source_src=" ".join(cache_src) + '\n'
                            else:
                                source_src = source.text.strip() + '\n'
                                cache_src = source.text.strip().split(' ')
                        if args.maxtokens and len(cache_src) > args.maxtokens:
                            continue
                        if args.mintokens and len(cache_src) < args.mintokens:
                            continue
                        source_n += 1
                        token_n += len(cache_src)
                        for target in sentence.findall('target'):
                            if target.get('langid')=='en':
                                if not target.text:
                                    if args.train:
                                        continue
                                    else:
                                        target.text = source_src
                                cache_trg = []
                                if tokenize:
                                    tokens_trg = nltk.word_tokenize(target.text.strip())
                                    for each in tokens_trg:
                                        cache_trg.append(each)
                                    target_trg=" ".join(cache_trg)+'\n'
                                else:
                                    target_trg = target.text.strip() + '\n'
                                    cache_trg = target.text.strip().split(' ')
                                if args.maxtokens and len(cache_trg) > args.maxtokens:
                                    continue
                                if args.mintokens and len(cache_trg) < args.mintokens:
                                    continue
                                if source_src == target_trg and not args.test:
                                    continue
                                ctx_file.write(source_ctx.encode('utf-8'))
                                token_src_ctx += cur_ctx_token
                                src_file.write(source_src.encode('utf-8'))
                                token_src_src += len(cache_src)
                                tag_file.write(target_trg.encode('utf-8'))
                                token_trg_trg += len(cache_trg)
                                count+=1
                        ctx_1 = ctx_2
                        ctx_2 = source_src.strip()
                        source_ctx = (ctx_1 + ' ' + ctx_2).strip() + '\n'
                        cur_ctx_token = len(source_ctx.strip().split(' '))
    print(args.input, ':', essay_n, 'essays,', source_n, 'source sentences,', token_n, 'tokens.')
    print('The number of source sentences / essays :', source_n / essay_n)
    print('The number of tokens / essays :', token_n / essay_n)
    if tokenize:
        print(count,'sentence pairs have been added with tokenization.')
    else:
        print(count,'sentence pairs have been added without tokenization.')
    print(src_ctx, token_src_ctx, 'tokens')
    print(src_src, token_src_src, 'tokens')
    print(trg_trg, token_trg_trg, 'tokens')


extract_from_xml(args.tokenize, args.input, args.src_ctx, args.src_src, args.trg_trg)




