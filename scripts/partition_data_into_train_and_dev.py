# This script partitions data into 2 sections, i.e., training data and development data
# Python version: 3.6.5
# Usage example for Lang-8: python partition_data_into_train_and_dev.py --dataset lang-8-20111007-L1-v2.xml --train lang8-train.xml --dev lang8-dev.xml --limit 3000
# Usage example for NUCLE: python partition_data_into_train_and_dev.py --dataset nucle3.2.xml --train nucle-train.xml --dev nucle-dev.xml --limit 5000 --m2 nucle3.2-preprocessed.conll.m2 --dev-m2 nucle-dev.raw.m2
import argparse
import xml.etree.cElementTree as ET

parser=argparse.ArgumentParser()
parser.add_argument('--dataset', help='dataset formatted by XML')
parser.add_argument('--train', help='selected train data from dataset')
parser.add_argument('--dev', help='selected development data from dataset')
parser.add_argument('--limit', type=int, help='the size of development data')
parser.add_argument('--m2', help='the m2 file of NUCLE dataset')
parser.add_argument('--dev-m2', help='the m2 file of the NUCLE development data')
args=parser.parse_args()

def partition_from_xml(dataset, train, dev, limit):
    if args.m2:
        with open(args.m2, encoding='UTF-8') as f:
            m2_file=f.readlines()
            pos = 0
            m2_dic={}
    dic_num={}
    dic_changes={}
    tree=ET.parse(dataset)
    root = tree.getroot()
    for essay in root:
        # if 'English' in essay.get('learning_language'):
        num=0
        changes=0
        if args.m2:
            m2_cache=[]
        for sentence in essay:
            source = sentence.find('source')
            if source.get('langid') != 'en':
                continue
            num += 1
            if args.m2:
                for i in range(pos,len(m2_file)):
                    if m2_file[i].startswith('S'):
                        assert m2_file[i].strip('\n')[2:] == source.text, 'match error!'
                    m2_cache.append(m2_file[i])
                    pos+=1
                    if m2_file[i]=='\n':
                        break
            for target in sentence.findall('target'):
                if target.get('langid') == 'en':
                    if not target.text:
                        continue
                    if source.text!=target.text:
                        changes+=1
                        break
        if args.m2:
            m2_dic[essay.get('id')]=m2_cache
        if num!=0:
            dic_num[essay.get('id')]=changes
            dic_changes[essay.get('id')]=changes/num

    rank_essay=sorted(dic_changes.items(), key=lambda dic_changes: dic_changes[1], reverse=True)
    train_list=[]
    dev_list=[]
    index=0
    size=0
    for each in rank_essay:
        if index%4==0 and size < limit:
            dev_list.append(each[0])
            size+=dic_num[each[0]]
        else:
            train_list.append(each[0])
        index+=1

    train_name=dataset.replace('.xml', '-train')
    train_data=ET.Element('dataset', attrib={'name': train_name})
    dev_name=dataset.replace('.xml', '-development')
    dev_data=ET.Element('dataset', attrib={'name': dev_name})
    dev_id=0
    train_id=0
    dev_m2=[]
    for essay in root:
        if essay.get('id') in dev_list:
            dev_data.append(essay)
            dev_m2.append(essay.get('id'))
            essay.set('id', str(dev_id))
            dev_id+=1
        else:
            train_data.append(essay)
            essay.set('id', str(train_id))
            train_id+=1
    indent(train_data)
    indent(dev_data)
    train_tree=ET.ElementTree(train_data)
    train_tree.write(train, encoding='UTF-8', xml_declaration=True)
    dev_tree=ET.ElementTree(dev_data)
    dev_tree.write(dev, encoding='UTF-8', xml_declaration=True)
    if args.m2:
        with open(args.dev_m2, 'w', encoding='UTF-8') as m2_output:
            for each in dev_m2:
                for line in m2_dic[each]:
                    m2_output.write(line)
    print('train data contains',len(train_list),'essays')
    print('development data contains', len(dev_list), 'essays')

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

partition_from_xml(args.dataset, args.train, args.dev, args.limit)