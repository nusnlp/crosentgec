# This script strips unchanged sentence pairs from M2 file.
# Python version: 3.6.5
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--nucle-dev', help='nucle-dev.src-trg.src')
parser.add_argument('--dev-m2', help='nucle-dev.raw.m2')
parser.add_argument('--processed-m2', help='The pre-processed m2 file of the NUCLE development data')
args = parser.parse_args()

source_sentences = 0
data = []
with open(args.nucle_dev, encoding='utf-8') as f:
    for line in f:
        data.append(line)
number = len(data)
with open(args.dev_m2, encoding='utf-8') as f1, open(args.processed_m2, 'w', encoding='utf-8') as f2:
    index = 0
    sign = 0
    for line in f1:
        if line.startswith('S'):
            if index < number and data[index] == line[2:]:
                sign = 1
                f2.write(line)
                source_sentences += 1
                index += 1
                continue
        if line.startswith('A'):
            if sign == 1:
                f2.write(line)
        else:
            if sign ==1:
                f2.write(line)
                sign = 0
print(source_sentences)
