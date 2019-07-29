# This file is part of the NUS M2 scorer.
# The NUS M2 scorer is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# The NUS M2 scorer is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# file: util.py
#

import sys
from .util import *



def load_annotation(gold_file, filter_etypes=['all']):
    source_sentences = []
    gold_edits = []
    fgold = smart_open(gold_file, 'r')
    puffer = fgold.read()
    fgold.close()
    #puffer = puffer.decode('utf8')
    for item in paragraphs(puffer.splitlines(True)):
        item = item.splitlines(False)
        sentence = [line[2:].strip() for line in item if line.startswith('S ')]
        assert sentence != []
        annotations = {}
        for line in item[1:]:
            if line.startswith('I ') or line.startswith('S '):
                continue
            assert line.startswith('A ')
            line = line[2:]
            fields = line.split('|||')
            start_offset = int(fields[0].split()[0])
            end_offset = int(fields[0].split()[1])
            etype = fields[1]
            if etype == 'noop':
                start_offset = -1
                end_offset = -1
            if "all" not in filter_etypes  and etype.lower() not in [filter_etype.lower() for filter_etype in filter_etypes]:
                continue
            corrections =  [c.strip() if c != '-NONE-' else '' for c in fields[2].split('||')]
            # NOTE: start and end are *token* offsets
            original = ' '.join(' '.join(sentence).split()[start_offset:end_offset])
            annotator = int(fields[5])
            if annotator not in annotations.keys():
                annotations[annotator] = []
            annotations[annotator].append((start_offset, end_offset, original, corrections, etype))
        tok_offset = 0
        for this_sentence in sentence:
            tok_offset += len(this_sentence.split())
            source_sentences.append(this_sentence)
            this_edits = {}
            for annotator, annotation in annotations.items():
                this_edits[annotator] = [edit for edit in annotation if edit[0] <= tok_offset and edit[1] <= tok_offset and edit[0] >= 0 and edit[1] >= 0]
            if len(this_edits) == 0:
                this_edits[0] = []
            gold_edits.append(this_edits)
    return (source_sentences, gold_edits)


def read_nbest_sentences(nbest_path):
    f_nbest = smart_open(nbest_path, 'r')
    index = 0
    nbest_sentences = []
    nbest_per_sentence = []
    for line in f_nbest:
        line = line.strip()
        pieces = line.split(' ||| ')
        if int(pieces[0]) == index:
            nbest_per_sentence.append(pieces[1])
        else:
            nbest_sentences.append(nbest_per_sentence)
            nbest_per_sentence = []
            nbest_per_sentence.append(pieces[1])
            index = index+1
    nbest_sentences.append(nbest_per_sentence)
    f_nbest.close()
    return nbest_sentences

def gold_to_m2(src, gold):			# has an ascii/utf-8 bug!
    m2_str = "S " + src + "\n"
    for annotator_id, annotations in gold.iteritems():
        for annotation in annotations:
           m2_str +=  "A "+ str(annotation[0]) + " " + str(annotation[1]) + "|||" + annotation[4] + "|||" + annotation[3][0]  + "|||REQUIRED|||-NONE-|||" + str(annotator_id) + "\n"
    m2_str += "\n"
    return m2_str



