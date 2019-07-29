# This script transforms Lang-8 data from Json format into XML format
# Python version: 3.6.5
# Usage example: python lang8_preprocess.py --dataset lang-8-20111007-L1-v2.dat --language English --id en --output lang-8-20111007-L1-v2.xml

import re,json
import langid
import argparse
import xml.etree.ElementTree as ET
from clean_data import clean_sentence

parser=argparse.ArgumentParser()
parser.add_argument('--dataset', help='raw lang8 dataset (json format)')
parser.add_argument('--language', help='the language that need be contained by the argument of learning language')
parser.add_argument('--id', help='the langid of this language')
parser.add_argument('--output', help='processed lang8 dataset (XML format)')
args=parser.parse_args()

# transform re to RegexObject
SLINE=re.compile("\[sline\].*?\[\\\/sline\]")
SLINE_END=re.compile("\[\\\/sline\]")
FTAG=re.compile("\[f-[a-zA-Z]*\]|\[\\\/f-[a-zA-Z]*\]")
BACKSLASH=re.compile(r'\\(.)')

def remove_tags(line):
    line.strip()
    for tag in SLINE,SLINE_END,FTAG:
        line=tag.sub('',line)
    return re.sub('\s+',' ',line)

def process(line,dataset,essay_id):
    changes=0
    row = json.loads(re.sub(r'[\x00-\x1F]+', '', line))
    if args.language != row[2]:
        return False, 0
    map=[]
    num=0
    match=0
    correction = False
    for i in range(len(row[4])):
        row[4][i] = re.sub('\s+',' ', row[4][i].strip())
        row[4][i] = clean_sentence(row[4][i])
        if len(row[4][i]):
            num+=1
            s_language, _ = langid.classify(row[4][i])
            if s_language==args.id:
                match+=1
                if correction == False:
                    for each in row[5][i]:
                        if each:
                            each = re.sub('\s+', ' ', each.strip())
                            each = clean_sentence(each)
                            if len(each):
                                t_language, _ = langid.classify(each)
                                if t_language == args.id and row[4][i] != each:
                                    correction = True
                                    break
            map.append(s_language)
        else:
            map.append('null')
    if match < 2 or correction == False:
        return False, 0
    essay=ET.SubElement(dataset, 'essay', attrib={'id': str(essay_id), 'journal_id':row[0], 'user_id':row[1], 'learning_language':row[2], 'native_language':row[3]})
    sentence_id = 0
    for i in range(len(row[4])):
        if len(row[4][i]):
            sentence=ET.SubElement(essay,'sentence', attrib={'id':str(sentence_id)})
            source=ET.SubElement(sentence,'source')
            source.text=row[4][i]
            source.set("langid",map[i])
            for each in row[5][i]:
                if each:
                    each = re.sub('\s+',' ', each.strip())
                    each = clean_sentence(each)
                    if len(each):
                        target=ET.SubElement(sentence,'target')
                        target.text=each
                        t_language, _ = langid.classify(target.text)
                        target.set("langid",t_language)
                        if t_language==args.id and source.text!=target.text:
                            changes+=1
            sentence_id += 1
    return True, changes

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

essay_id=0
dataset=ET.Element('dataset',attrib={'name':'lang-8-20111007-L1-v2'})
with open(args.dataset, encoding='utf-8') as f:
    size=0
    for line in f:
        line=remove_tags(line)
        judge, changes=process(line,dataset,essay_id)
        if judge==True:
            essay_id+=1
        size+=changes
    indent(dataset)
    tree = ET.ElementTree(dataset)
    tree.write(args.output, encoding='UTF-8', xml_declaration=True)
    print(essay_id, "essays have been added.")
    print(size, "sentence pairs can be used.")