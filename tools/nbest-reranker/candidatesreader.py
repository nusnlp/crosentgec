import sys
#import dlm.utils as U
#import dlm.io.logging as L
import codecs

class NBestList():
    def __init__(self, nbest_path, mode='r', reference_list=None):
        assert mode == 'r' or mode == 'w', "Invalid mode: " + mode
        self.mode = mode
        self.nbest_file = codecs.open(nbest_path, mode=mode, encoding='UTF-8')
        self.prev_index = -1
        self.curr_item = None
        self.curr_index = 0
        self.eof_flag = False
        self.ref_manager = None
        if reference_list:
            assert mode == 'r', "Cannot accept a reference_list in 'w' mode"
            self.ref_manager = RefernceManager(reference_list)


    def __iter__(self):
        assert self.mode == 'r', "Iteration can only be done in 'r' mode"
        return self



    def __next__(self): # Returns a group of NBestItems with the same index
        if self.eof_flag == True:
            raise StopIteration
        assert self.mode == 'r', "next_group() method can only be used in 'r' mode"
        group = NBestGroup(self.ref_manager)
        group.add(self.curr_item) # add the item that was read in the last next() call
        try:
            self.curr_item = self.next_item()
        except StopIteration:
            self.eof_flag = True
            return group
        if self.curr_index != self.curr_item.index:
            self.curr_index = self.curr_item.index
            return group
        while self.curr_index == self.curr_item.index:
            group.add(self.curr_item)
            try:
                self.curr_item = self.next_item()
            except StopIteration:
                self.eof_flag = True
                return group
        self.curr_index = self.curr_item.index
        return group

    next = __next__  # Python 2


    def next_item(self):
        assert self.mode == 'r', "next() method can only be used in 'r' mode"
        try:
            if sys.version_info[0] >= 3:
                segments = self.nbest_file.__next__().split("|||")
            else:
                segments = self.nbest_file.next().split("|||")
        except StopIteration:
            self.close()
            raise StopIteration
        try:
            index = int(segments[0])
        except ValueError:
                    print >> sys.stderr, "The first segment in an n-best list must be an integer"
            #L.error("The first segment in an n-best list must be an integer")
        hyp = segments[1].strip()
        features = segments[2].strip()
        score = None
        phrase_alignments = None
        word_alignments = None
        phrase_alignments = None
        if len(segments) > 3:
            score = segments[3].strip()
        if len(segments) > 4:
            phrase_alignments = segments[4].strip()
        if len(segments) > 5:
            word_alignments = segments[5].strip()
        return NBestItem(index, hyp, features, score, phrase_alignments, word_alignments)


    def write(self, item):
        assert self.mode == 'w', "write() method can only be used in 'w' mode"
        self.nbest_file.write(str(item) + "\n")

    def close(self):
        self.nbest_file.close()




class NBestItem:
    def __init__(self, index, hyp, features, score, phrase_alignments, word_alignments):
        self.index = index
        self.hyp = hyp
        self.features = features
        self.score = score
        self.phrase_alignments = phrase_alignments
        self.word_alignments = word_alignments

    def __str__(self):
        output = ' ||| '.join([str(self.index), self.hyp, self.features])
        if self.score:
            output = output + ' ||| ' + self.score
        if self.phrase_alignments:
            output = output + ' ||| ' + self.phrase_alignments
        if self.word_alignments:
            output = output + ' ||| ' + self.word_alignments
        return output

    def append_feature(self, feature_name, feature_value):
        self.features += ' ' + str(feature_name) + '= ' + str(feature_value) + ' '


class NBestGroup:
    def __init__(self, refrence_manager=None):
        self.group_index = -1
        self.group = []
        self.ref_manager = refrence_manager

    def __str__(self):
        return '\n'.join([str(item) for item in self.group])

    def __iter__(self):
        self.item_index = 0
        return self

    def __getitem__(self, index):
        return self.group[index]

    def add(self, item):
        if item is None:
            return
        if self.group_index == -1:
            self.group_index = item.index
            if self.ref_manager:
                self.refs = self.ref_manager.get_all_refs(self.group_index)
        else:
            assert item.index == self.group_index, "Cannot add an nbest item with an incompatible index"
        self.group.append(item)

    def __next__(self):
        #if self.item_index < len(self.group):
        try:
            item = self.group[self.item_index]
            self.item_index += 1
            return item
        #else:
        except IndexError:
            raise StopIteration

    def size(self):
        return len(self.group)

    def append_features(self, features_list):
        assert len(features_list) == len(self.group), 'Number of features and number of items in this group do not match'
        for i in range(len(self.group)):
            self.group[i].append_feature(features_list[i])

    next = __next__  # Python 2


class RefernceManager:
    def __init__(self, paths_list):
        assert type(paths_list) is list, "The input to a RefernceManager class must be a list"
        self.ref_list = []
        self.num_lines = -1
        self.num_refs = 0
        for path in paths_list:
            with codecs.open(path, mode='r', encoding='UTF-8') as f:
                self.num_refs += 1
                sentences = f.readlines()
                if self.num_lines == -1:
                    self.num_lines = len(sentences)
                else:
                    assert self.num_lines == len(sentences), "Reference files must have the same number of lines"
                self.ref_list.append(sentences)

    def get_all_refs(self, index):
        assert index < self.num_lines, "Index out of bound"
        return [self.ref_list[k][index] for k in range(self.num_refs)]





























