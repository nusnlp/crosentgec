# This script generates XML file for NUCLE data. Besides, it can also be used to preprocess CoNLL-2013 and CoNLL-2014 data.
# Python version: 3.6.5
# Usage python nucle_preprocess.py conllFileName m2FileName XMLFileName

import sys
import xml.etree.ElementTree as ET

def essay_boundary(file):
    boundary=[]
    essay=[]
    sentence=[]
    essay_id=''
    space=0
    i=0
    with open(file,encoding='utf-8') as f:
        for line in f:
            cache=line.strip().split('\t')
            if cache[0]:
                if essay_id!=cache[0]:
                    essay_id=cache[0]
                    essay.append(essay_id)
                    sentence.append(cache[4])
                    space=0
                else:
                    if not space:
                        sentence.append(cache[4])
            else:
                space=1
                if essay:
                    essay.append(" ".join(sentence))
                    boundary.append(essay)
                essay=[]
                sentence=[]
    return boundary

def src_tag_map(file):
    nucle=[]
    src_tag=[]
    words=[]
    corrected = []
    sid = eid = 0
    prev_sid = prev_eid = -1
    pos = 0
    with open(file,encoding='utf-8') as input_file:
        for line in input_file:
            line = line.strip()
            if line.startswith('S'):
                line = line[2:]
                words = line.split()
                corrected = ['<S>'] + words[:]
                src_tag.append(line)
            elif line.startswith('A'):
                line = line[2:]
                info = line.split("|||")
                sid, eid = info[0].split()
                sid = int(sid) + 1; eid = int(eid) + 1
                error_type = info[1]
                if error_type == "Um":
                    continue
                for idx in range(sid, eid):
                    corrected[idx] = ""
                if sid == eid:
                    if sid == 0: continue
                    if sid != prev_sid or eid != prev_eid:
                        pos = len(corrected[sid-1].split())
                    cur_words = corrected[sid-1].split()
                    cur_words.insert(pos, info[2])
                    pos += len(info[2].split())
                    corrected[sid-1] = " ".join(cur_words)
                else:
                    corrected[sid] = info[2]
                    pos = 0
                prev_sid = sid
                prev_eid = eid
            else:
                target_sentence = ' '.join([word for word in corrected if word != ""])
                assert target_sentence.startswith('<S>'), '(' + target_sentence + ')'
                target_sentence = target_sentence[4:]
                src_tag.append(target_sentence)
                nucle.append(src_tag)
                src_tag=[]
                prev_sid = -1
                prev_eid = -1
                pos = 0
    return nucle

def generate_XMLTree(root, conll_file, m2_file):
    boundary=essay_boundary(conll_file)
    nucle=src_tag_map(m2_file)
    i=0
    sentence_id=0
    essay=ET.SubElement(root,'essay',attrib={'id': str(i), 'journal_id': boundary[i][0], 'user_id': 'N.A', 'learning_language': 'English', 'native_language': 'N.A'})
    for each in nucle:
        if each[0]==boundary[i][1]:
            if i:
                essay=ET.SubElement(root,'essay',attrib={'id': str(i), 'journal_id': boundary[i][0], 'user_id': 'N.A', 'learning_language': 'English', 'native_language': 'N.A'})
                sentence_id=0
            if(i<len(boundary)-1):
                i+=1
        sentence = ET.SubElement(essay, 'sentence', attrib={'id': str(sentence_id)})
        source=ET.SubElement(sentence,'source',attrib={'langid': 'en'})
        source.text=each[0]
        target=ET.SubElement(sentence,'target',attrib={'langid': 'en'})
        target.text=each[1]
        sentence_id+=1

def indent(elem, level=0):
  i = "\n" + level*"  "
  if len(elem):
    if not elem.text or not elem.text.strip():
      elem.text = i + "  "
    if not elem.tail or not elem.tail.strip():
      elem.tail = i
    for elem in elem:
      indent(elem, level+1)
    if not elem.tail or not elem.tail.strip():
      elem.tail = i
  else:
    if level and (not elem.tail or not elem.tail.strip()):
      elem.tail = i

if len(sys.argv)!=4:
    print("[USAGE] %s  conll_file  m2_file  xml_file" % sys.argv[0])
    sys.exit()

conll_file=sys.argv[1]
m2_file=sys.argv[2]
xml_file=sys.argv[3]

dataset=ET.Element('dataset',attrib={'name':xml_file})
generate_XMLTree(dataset,conll_file,m2_file)
indent(dataset)
tree=ET.ElementTree(dataset)
tree.write(xml_file, encoding='UTF-8', xml_declaration=True)

