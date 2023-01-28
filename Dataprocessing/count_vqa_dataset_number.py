import json
import os

SCANVQA_TRAIN = []
# SCANVQA_TRAIN = json.load(open(os.path.join("data", "ScanVQA/ScanVQA_train.json")))
# SCANVQA_TRAIN += json.load(open(os.path.join("data", "ScanVQA/ScanVQA_train.json")))
# SCANVQA_TRAIN += json.load(open(os.path.join("data", "ScanVQA/ScanVQA_train.json")))
# SCANVQA_TRAIN += json.load(open(os.path.join("data", "ScanVQA/ScanVQA_train.json")))
## TODO more "data"set
SCANVQA_TRAIN += json.load(open(os.path.join("data", "ScanVQA/ScanVQA_generated.json")))
# SCANVQA_TRAIN += json.load(open(os.path.join("data", "ScanVQA/ScanRefer_filtered_generated.json")))
# SCANVQA_TRAIN += json.load(open(os.path.join("data", "ScanVQA/nr3d_generated.json")))
# 
# # # TODO template-based; "data"set-size x 2
# #SCANVQA_TRAIN = json.load(open(os.path.join("data", "ScanVQA/ScanVQA_train.json")))
# #SCANVQA_TRAIN += json.load(open(os.path.join("data", "ScanVQA/ScanVQA_generated.json")))
# SCANVQA_TRAIN += json.load(open(os.path.join("data", "ScanVQA/ScanRefer_filtered_generated_masked.json")))
# SCANVQA_TRAIN += json.load(open(os.path.join("data", "ScanVQA/nr3d_generated_masked.json")))

SCANVQA_VAL = json.load(open(os.path.join("data", "ScanVQA/ScanVQA_train.json")))  # UNSEEN
# SCANVQA_VAL = json.load(open(os.path.join("data", "ScanVQA/ScanVQA_val.json")))  # UNSEEN
# SCANVQA_VAL_SEEN = json.load(open(os.path.join(""data"", "ScanVQA_val.json")))

def check_number(SCANVQA_TRAIN, item_name=None):
    number = 0
    for items in SCANVQA_TRAIN:
        if item_name is not None:
            if item_name in items.keys():
                number += len(items[item_name])
            continue
        if 'related_object(type 1)' in items:
            number += len(items['related_object(type 1)'])
        if 'related_object(type 2)' in items:
            number += len(items['related_object(type 2)'])
        if 'related_object(type 3)' in items:
            number += len(items['related_object(type 3)'])
    print(item_name, len(SCANVQA_TRAIN), number, 1. * number / len(SCANVQA_TRAIN))

check_number(SCANVQA_TRAIN)
check_number(SCANVQA_TRAIN, 'related_object(type 1)')
check_number(SCANVQA_TRAIN, 'related_object(type 2)')
check_number(SCANVQA_TRAIN, 'related_object(type 3)')
