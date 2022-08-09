import os, json
import matplotlib.pyplot as plt

data_root = '../data/ECOB-ZH'
train_doc_file = os.path.join(data_root, 'train.doc.json')
train_ann_file = os.path.join(data_root, 'train.ann.json')
dev_doc_file = os.path.join(data_root, 'dev.doc.json')
dev_ann_file = os.path.join(data_root, 'dev.ann.json')
test_doc_file = os.path.join(data_root, 'test.doc.json')
test_ann_file = os.path.join(data_root, 'test.ann.json')

def loadJson(filename):
    f = open(filename, 'r', encoding='utf-8')
    res = json.load(f)
    # print(res)
    f.close()
    return res

def showLenContribution(len_arr, commit=''):
    plt.figure(figsize=(5, 4))
    x = [i for i in range(len(len_arr))]
    y = len_arr
    plt.plot(x, y)
    plt.xlabel('length')
    plt.ylabel('count')
    plt.legend(title=commit)
    plt.show()
    return 0

def get1DMatrixInfo(matrix, showContribution=False, maxlen=512, start_idx=0, commit=''):
    max = 0
    min = 99999999
    sum = 0
    min_idx = 0
    max_idx = 0
    len_arr = [0 for i in range(maxlen)]
    for i in range(0, len(matrix)):
        sum += matrix[i]
        len_arr[matrix[i]] += 1
        if matrix[i] > max:
            max = matrix[i]
            max_idx = i
        if matrix[i] < min:
            min = matrix[i]
            min_idx = i
    if showContribution:
        showLenContribution(len_arr, commit)
    return {
        'data': commit,
        'max': max,
        'min': min,
        'avg': sum / len(matrix),
        'max_idx': max_idx + start_idx,
        'min_idx': min_idx + start_idx,
        'len_contribution': len_arr if showContribution else 'unshow'
    }

def get2DMatrixInfo(matrix, showContribution=False, maxlen=512, start_idx=0, commit=''):
    max = 0
    min = 99999999
    sum = 0
    item_count = 0
    min_idx = [0, 0]
    max_idx = [0, 0]
    len_arr = [0 for i in range(maxlen)]
    for i in range(0, len(matrix)):
        for j in range(0, len(matrix[i])):
            sum += matrix[i][j]
            len_arr[matrix[i][j]] += 1
            item_count += 1
            if matrix[i][j] > max:
                max = matrix[i][j]
                max_idx = [i, j]
            if matrix[i][j] < min:
                min = matrix[i][j]
                min_idx = [i, j]
    if showContribution:
        showLenContribution(len_arr, commit)
    return {
        'data': commit,
        'max': max,
        'min': min,
        'avg': sum / item_count,
        'max_idx': [max_idx[0] + start_idx,  max_idx[1]],
        'min_idx': [min_idx[0] + start_idx,  min_idx[1]],
        'len_contribution': len_arr if showContribution else 'unshow'
    }

class dataAnalyzer:
    def __init__(self, doc_file='', ann_file='', start_doc_idx=0, start_event_idx=0):
        self.doc_list = loadJson(doc_file)
        self.ann_list = loadJson(ann_file)
        self.start_doc_idx = start_doc_idx
        self.start_event_idx = start_event_idx
        self.matrix = self.json2matrix()
        self.doc_sentence_len_matrix = self.getDocSentenceLenMatrix()
        self.doc_len_matrix = self.getDocLenMatrix()
        self.doc_sentence_num_matrix = self.getDocSentenceNumMatrix()
        
        self.descriptor_len_matrix = self.getDescriptorLenMatrix()

        self.viewpoint_len_matrix = self.getViewpointLenMatrix()
        self.viewpoint_sentence_num_matrix = self.getViewpointSentenceNumMatrix()

    def json2matrix(self):
        doc_list = self.doc_list
        ann_list = self.ann_list
        matrix = []
        for i in range(len(doc_list)):
            matrix.append([[-1, -1]])
        for ann in ann_list:
            doc_id = ann['doc_id'] - self.start_doc_idx
            start_sent_idx = ann['start_sent_idx']
            end_sent_idx = ann['end_sent_idx']
            doc_row = matrix[doc_id]
            doc_row[0][0] = ann['event_id']
            #doc_row[0][1] = len(doc_list[doc_id]['Descriptor']['text'])
            current_not_view_idx = doc_row[0][len(doc_row[0]) - 1] + 1
            doc_content_list = doc_list[doc_id]['Doc']['content']
            for idx in range(current_not_view_idx, start_sent_idx):
                doc_row.append([len(doc_content_list[idx]['sent_text']) ,False])
            view_sentence_len_list = []
            for idx in range(start_sent_idx, end_sent_idx + 1):
                view_sentence_len_list.append(len(doc_content_list[idx]['sent_text']))
            view_sentence_len_list.append(True)
            doc_row.append(view_sentence_len_list)
            doc_row[0][len(doc_row[0]) - 1] = end_sent_idx
        for doc_id in range(len(doc_list)):
            doc_row = matrix[doc_id]
            doc_content_list = doc_list[doc_id]['Doc']['content']
            for idx in range(doc_row[0][len(doc_row[0]) - 1] + 1, len(doc_content_list)):
                doc_row.append([len(doc_content_list[idx]['sent_text']) ,False])
            doc_row[0][len(doc_row[0]) - 1] = len(doc_content_list)
        return matrix

    def getDocSentenceLenMatrix(self):
        matrix = self.matrix
        doc_sentence_len_matrix = []
        for doc_id in range(len(matrix)):
            doc_sentence_len_row = []
            doc_row = matrix[doc_id]
            for i in range(1, len(doc_row)):
                for j in range(0, len(doc_row[i]) - 1):
                    doc_sentence_len_row.append(doc_row[i][j])
            doc_sentence_len_matrix.append(doc_sentence_len_row)
        return doc_sentence_len_matrix

    def getDocLenMatrix(self):
        doc_sentence_len_matrix = self.doc_sentence_len_matrix
        doc_len_matrix = []
        for doc_id in range(len(doc_sentence_len_matrix)):
            doc_len_matrix.append(sum(doc_sentence_len_matrix[doc_id]))
        return doc_len_matrix

    def getDocSentenceNumMatrix(self):
        doc_sentence_len_matrix = self.doc_sentence_len_matrix
        doc_sentence_num_matrix = []
        for doc_id in range(len(doc_sentence_len_matrix)):
            doc_sentence_num_matrix.append(len(doc_sentence_len_matrix[doc_id]))
        return doc_sentence_num_matrix

    def getDescriptorLenMatrix(self):
        doc_list = self.doc_list
        event_cnt = doc_list[len(doc_list) - 1]['Descriptor']['event_id'] + 1 - self.start_event_idx
        descriptor_len_matrix = []
        for i in range(event_cnt):
            descriptor_len_matrix.append(0)
        for doc_id in range(len(doc_list)):
            descriptor = doc_list[doc_id]['Descriptor']
            descriptor_len_matrix[descriptor['event_id'] - self.start_event_idx] = len(descriptor['text'])
        return descriptor_len_matrix

    def getViewpointLenMatrix(self):
        matrix = self.matrix
        viewpoint_len_matrix = []
        for doc_id in range(len(matrix)):
            viewpoint_len_row = []
            doc_row = matrix[doc_id]
            for i in range(1, len(doc_row)):
                if doc_row[i][len(doc_row[i]) - 1]:
                    sum = 0
                    for j in range(0, len(doc_row[i]) - 1):
                        sum += doc_row[i][j]
                    viewpoint_len_row.append(sum)
            viewpoint_len_matrix.append(viewpoint_len_row)
        return viewpoint_len_matrix

    def getViewpointSentenceNumMatrix(self):
        matrix = self.matrix
        viewpoint_sentence_num_matrix = []
        for doc_id in range(len(matrix)):
            viewpoint_sentence_num_row = []
            doc_row = matrix[doc_id]
            for i in range(1, len(doc_row)):
                if doc_row[i][len(doc_row[i]) - 1]:
                    viewpoint_sentence_num_row.append(len(doc_row[i]) - 1)
            viewpoint_sentence_num_matrix.append(viewpoint_sentence_num_row)
        return viewpoint_sentence_num_matrix

    def getSentenceLenInfo(self, showContribution=False, maxlen=512):
        return get2DMatrixInfo(self.doc_sentence_len_matrix, showContribution, maxlen, self.start_doc_idx, 'sentence length(char level)')

    def getDocLenInfo(self, showContribution=False, maxlen=512):
        return get1DMatrixInfo(self.doc_len_matrix, showContribution, maxlen, self.start_doc_idx, 'doc length(char level)')

    def getDocSentenceNumInfo(self, showContribution=False, maxlen=512):
        return get1DMatrixInfo(self.doc_sentence_num_matrix, showContribution, maxlen, self.start_doc_idx, 'doc length(sentence level)')
    
    def getDescriptorLenInfo(self, showContribution=False, maxlen=512):
        return get1DMatrixInfo(self.descriptor_len_matrix, showContribution, maxlen, self.start_event_idx, 'descriptor length(char level)')

    def getViewpointLenInfo(self, showContribution=False, maxlen=512):
        return get2DMatrixInfo(self.viewpoint_len_matrix, showContribution, maxlen, self.start_doc_idx, 'viewpoint length(char level)')

    def getViewpointSentenceNumInfo(self, showContribution=False, maxlen=512):
        return get2DMatrixInfo(self.viewpoint_sentence_num_matrix, showContribution, maxlen, self.start_doc_idx, 'viewpoint length(sentence level)')


def showResult():
    trainAnalyzer = dataAnalyzer(test_doc_file, test_ann_file, start_doc_idx=2401, start_event_idx=668)

    print("dataset: test")
    print(trainAnalyzer.getSentenceLenInfo(showContribution=True, maxlen=512))
    print(trainAnalyzer.getDocLenInfo(showContribution=False, maxlen=5120))
    print(trainAnalyzer.getDocSentenceNumInfo(showContribution=True, maxlen=50))
    print(trainAnalyzer.getDescriptorLenInfo(showContribution=True, maxlen=100))
    print(trainAnalyzer.getViewpointLenInfo(showContribution=False, maxlen=5120))
    print(trainAnalyzer.getViewpointSentenceNumInfo(showContribution=True, maxlen=50))

if __name__ == '__main__':
    showResult()