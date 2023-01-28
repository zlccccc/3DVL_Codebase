import os
import sys
import json

import xlrd
import re
import numpy as np

sys.path.append(os.path.join(os.getcwd()))  # HACK add the root folder
from data.scannet.model_util_scannet import rotate_aligned_boxes, ScannetDatasetConfig, rotate_aligned_boxes_along_axis
DC = ScannetDatasetConfig()

number = 'zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen ' \
         'seventeen eighteen nineteen twenty'.split(' ')
def detection_based():
    output_file = os.path.join("./data/ScanVQA/", "ScanVQA_generated.json")
    filename = "data/scannet_detection.xlsx"
    excel = xlrd.open_workbook(filename)
    table = excel.sheets()[0]
    entries = []
    rows = table.nrows  # 获取行数
    cols = table.ncols  # 获取列数; 应该是10
    # titles = ['scene_id', 'question_type', 'question', 'answer',
    #           'Grounding in Query', 'Contextual Object of Grounding',
    #           'Grounding in Answer', 'related_object(type 4)', # no type 4
    #           'rank(filter)', 'issue(filter)']
    organized, obj_names, scene_count = [], table.row_values(0), 0
    for i in range(1, rows):
        line = table.row_values(i)
        filename = line[0]
        if '_00' not in filename:
            continue
        scene_count += 1
        # print(filename)
        current_label = {
            'source': 'detection based',
            'scene_id': filename,
            'question_type': 'what is',
            'question': 'what is this room?',
            'answer': line[1].split('.')[0].lower(),
            'Grounding in Query': [],  # todo
            'Contextual Object of Grounding': [],  # todo
            'Grounding in Answer': [],  # todo
            # 'related_object(type 4)': [],  # todo
            'rank(filter)': 'A',
            'issue(filter)': 'template based'
        }
        instance_bboxes = np.load(os.path.join('./data/scannet/scannet_data', filename)+"_aligned_bbox.npy")
        organized.append(current_label)
        for k in range(2, cols):
            type = obj_names[k].split('(')[1].split(')')[0]
            # type_id = int(obj_names[k].split('(')[0])
            if type == 'otherfurniture':
                type = 'others'
                continue  # removed
            type_id = DC.type2class[type]
            obj_label = []
            # print(instance_bboxes.shape)
            for ins in instance_bboxes:
                # print(ins, type)
                if DC.nyu40id2class[int(ins[-2])] == type_id:
                    obj_label.append(int(ins[-1]))
                    pass
                # print(t)
            # print(len(obj_label), int(line[k]), 'type', type, type_id, DC.class2type[type_id])
            # print(type)
            if type == 'other':
                type = 'otherfurniture'
            if int(line[k]) < 10:
                # todo type3
                current_label = {
                    'source': 'detection based',
                    'scene_id': filename,
                    'question_type': 'how many',
                    'question': f'how many {type} are there this room?',
                    'answer': number[int(line[k])],
                    'Grounding in Query': obj_label,  # todo
                    'Contextual Object of Grounding': [],  # todo
                    'Grounding in Answer': [],  # todo
                    # 'related_object(type 4)': [],  # todo
                    'rank(filter)': 'A',
                    'issue(filter)': 'template based'
                }
                # todo type1
                organized.append(current_label)
                question = [f'is there a {type} in this room?', f'is there any {type} in this room?', f'is there {type}s in this room?']
                current_label = {
                    'source': 'detection based',
                    'scene_id': filename,
                    'question_type': 'is there',
                    'question': np.random.choice(question),
                    'answer': 'yes' if int(line[k]) else 'no',
                    'Grounding in Query': obj_label,  # todo
                    'Contextual Object of Grounding': [],  # todo
                    'Grounding in Answer': [],  # todo
                    # 'related_object(type 4)': [],  # todo
                    'rank(filter)': 'A',
                    'issue(filter)': 'template based'
                }
                organized.append(current_label)
                # break
    with open(output_file, "w") as f:
        json.dump(organized, f, indent=4)
        print('detection based: {} QAs in {} scenes processed'.format(len(organized), scene_count))
        # TODO change TYPE1


def relation_based(prefix, mask=False):
    # TODO: relation-based data
    if mask:
        output_file = os.path.join("./data/ScanVQA/", f"{prefix}_generated_masked.json")
    else:
        output_file = os.path.join("./data/ScanVQA/", f"{prefix}_generated.json")
    organized = []
    scenes = set()
    TRAIN = json.load(open(os.path.join('./data/', f"{prefix}_train.json")))
    VAL = json.load(open(os.path.join('./data/', f"{prefix}_val.json")))  # UNSEEN
    for value in TRAIN+VAL:
        # print(value)
        # 'object_id': value['object_id']
        scenes.add(value['scene_id'])
        if not mask:
            current_label = {
                'source': f'{prefix} dataset based',
                'scene_id': value['scene_id'],
                'question_type': 'grounding',
                'question': value['description'],
                'answer': ' '.join(value['object_name'].split('_')),
                'Grounding in Query': [int(value['object_id'])],  # todo
                # 'Grounding in Answer': [],  # todo
                'rank(filter)': 'A',
                'issue(filter)': 'template based'
            }
            organized.append(current_label)
        if mask:
            current_label = {
                'source': f'{prefix} dataset based',
                'scene_id': value['scene_id'],
                'question_type': 'grounding',
                'question': value['description'].replace(value['object_name'], '[mask]'),
                'answer': ' '.join(value['object_name'].split('_')),
                # 'Grounding in Query': [],  # todo
                'Grounding in Answer': [int(value['object_id'])],  # todo
                'rank(filter)': 'A',
                'issue(filter)': 'template based'
            }
            organized.append(current_label)
        # break
    with open(output_file, "w") as f:
        json.dump(organized, f, indent=4)
    print('relation based: {} QAs in {} scenes processed, mask={}'.format(len(organized), len(scenes), mask))


detection_based()
relation_based('ScanRefer_filtered')
relation_based('nr3d')
relation_based('ScanRefer_filtered', True)
relation_based('nr3d', True)
