import json
import numpy


def get_doc_ann(ann_file_name):
    with open(ann_file_name, 'r', encoding='utf-8') as f:
        ann_list = json.load(f)
        doc_ann = {}
        for ann in ann_list:
            doc_ann[ann['doc_id']] = []
        for ann in ann_list:
            doc_ann[ann['doc_id']].append((ann['start_sent_idx'], ann['end_sent_idx']))
        return doc_ann


def train_dev(doc_file, ann_file, max_seq_lens):
    descriptors = []
    contents = []
    start_anns = []
    end_anns = []
    valid_lens = []
    doc_ann = get_doc_ann(ann_file)
    with open(doc_file, 'r', encoding='utf-8') as f_doc:
        doc_list = json.load(f_doc)
        for doc in doc_list:
            Descriptor = doc['Descriptor']
            text = Descriptor['text']
            descriptors.append(text)
            Doc = doc['Doc']
            doc_id = Doc['doc_id']
            # print(doc_id)
            content = Doc['content']
            content_text = [c['sent_text'] for c in content]
            padding_context = [''] * (max_seq_lens - len(content))
            content_text[len(content):] = padding_context
            start_ann = numpy.array([0] * len(content_text))
            end_ann = numpy.array([0] * len(content_text))
            ann_list = doc_ann.get(doc_id)
            if ann_list is not None:
                for a in ann_list:
                    start_ann[a[0]] = 1
                    end_ann[a[1]] = 1
            valid_lens.append(len(content))
            contents.append(content_text)
            start_anns.append(start_ann)
            end_anns.append(end_ann)
    return descriptors, contents, start_anns, end_anns, valid_lens

