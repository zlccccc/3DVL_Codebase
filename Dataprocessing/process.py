import os
import sys
import json

import xlrd
import re

sys.path.append(os.path.join(os.getcwd()))  # HACK add the root folder


# 获取场景内标注, 输出json
def solve(file_base_path, output_file, filter=False):
    organized = []
    print("parsing...")
    scene_count = 0
    scene_ids = []
    # 在本地文件夹下建立一个file文件; 会遍历那个file文件夹
    # print('titles not right是文件第一行长度不够长')
    # print('float no attribute lower是在回答时,回答了1234...的阿拉伯数字')
    number = 'zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen ' \
             'seventeen eighteen nineteen twenty'.split(' ')
    f_out = open('log.txt', 'w')
    # print(number)
    # for root, dirs, files in os.walk(".\\Filtered_v1", topdown=False):
    for root, dirs, files in os.walk(file_base_path, topdown=False):
        for name in files:
            filename = os.path.join(root, name)
            if '.xlsx' not in filename:
                continue
            try:
                scene_count += 1
                scene_ids.append([int(name[6:9]), 0])
                now_scene_id = name[0:12]
                # if '0000' not in filename:  # 测试
                #     continue
                # 读取工作表
                excel = xlrd.open_workbook(filename)
                # table = excel.sheet_by_name('工作表1')
                table = excel.sheets()[0]
                entries = []
                rows = table.nrows  # 获取行数
                cols = table.ncols  # 获取列数; 应该是10
                if cols != 10:
                    print(filename, 'cols != 10; titles are not right (title长度不对)')
                # print(rows, cols, 'in', filename)
                # print(table.row_values(0))  # titles;
                titles = ['scene_id', 'question_type', 'question', 'answer',
                          'Grounding in Query', 'Contextual Object of Grounding',
                          'Grounding in Answer', 'related_object(type 4)',
                          'rank(filter)', 'issue(filter)']

                scene_id_not_right = False  # 检查scene_id是否正确，如果错误只输出一次
                for i in range(1, rows):
                    line = table.row_values(i)
                    if filter and line[8] == '':
                        continue
                    if "".join([str(item) for item in line[1:]]) == "":  # excel中存在一些别的东西
                        print("文件", filename, "处理错误, 存在空行", i + 1)
                        break
                    if line[0] != now_scene_id:
                        if scene_id_not_right:
                            print("文件", filename, "第", i + 1, "行 scene id 错误", "现为", line[0], "应为", now_scene_id)
                            scene_id_not_right = True
                        line[0] = now_scene_id
                    entry = {}
                    # break
                    qa_str_not_right = False
                    for j, name in enumerate(titles):
                        if j >= cols:
                            value_str = ""
                        else:
                            value_str = line[j]
                        if 'related_object' in name:
                            if isinstance(value_str, float):
                                value_str = str(int(value_str))
                            value_str = re.split('[，,.。？? ]', value_str)
                            value_str = [int(value) for value in value_str if value != ""]
                        if name in ['question', 'answer', 'question_type']:
                            if isinstance(value_str, float):  # 回答了阿拉伯数字; do not remove
                                value_str = str(int(value_str)).lower()
                                print(now_scene_id, 'line {}'.format(i + 1), '回答了阿拉伯数字, 已处理')
                                print(line)
                            value_str = value_str.strip(' ').strip(' ')
                            value_str = value_str.replace('\xa0', ' ')
                            value_str = value_str.lower()
                            if value_str == "":
                                qa_str_not_right = True
                        entry[name] = value_str
                        # print(line)
                    if qa_str_not_right:
                        print(filename, "line {} question或answer为空".format(i + 1))
                        continue
                    entries.append(entry)
                # print('file', filename, len(entries), 'entries')
                print(now_scene_id, len(entries), file=f_out)
                # print(len(entries), file=f_out)
                organized.extend(entries)
                scene_ids[-1][1] = len(entries)
                # break
            except Exception as e:
                print(filename, e, '存在问题; throw an Exception')

        for name in dirs:
            # print(os.path.join(root, name))
            pass

    with open(output_file, "w") as f:
        json.dump(organized, f, indent=4)

    print('{} QAs in {} scenes processed'.format(len(organized), scene_count))
    # for i in range(0, 706, 100):
    #     print('{} scenes in range({}, {}), QA number {}'.format(
    #         len([x for x in scene_ids if (i <= x[0] < i + 100)]), i, i + 100,
    #         sum([x[1] for x in scene_ids if (i <= x[0] < i + 100)])))
    print("done!")

    scene_id_bak = []
    for value in scene_ids:
        value = value[0]
        if value in scene_id_bak:
            print(value, 'solved twice!')
        scene_id_bak.append(value)
    return organized, scene_ids

# 获取scene和人员的匹配
def get_correspondence(filename):
    excel = xlrd.open_workbook(filename)
    table = excel.sheets()[1]
    rows = table.nrows  # 获取行数
    cols = table.ncols  # 获取列数; 应该是10
    scene_ids = []
    correspondence = {
        '标注': {},
        '筛选': {}
    }
    for i in range(1, 708):
        line = table.row_values(i)
        if line[2]:
            # print(line[0], line[2], line[4], line[5])
            if line[4]:
                scene_ids.append([int(line[0][6:9]), int(line[4])])
        if 'scene' in line[0]:
            correspondence['标注'][int(line[0][6:9])] = line[2]
            correspondence['筛选'][int(line[0][6:9])] = line[5]

    return correspondence, scene_ids

# 输出scene标了多少个
def output_by_scene(scene_ids, prefix):
    for i in range(0, 706, 100):
        print('{}: {} scenes in range({}, {}), QA number {}'.format(prefix,
            len([x for x in scene_ids if (i <= x[0] < i + 100) and x[1]]), i, i + 100,
            sum([x[1] for x in scene_ids if (i <= x[0] < i + 100)])))
    print('{} QA in the scenes.'.format(sum([x[1] for x in scene_ids])))


# 获取每个人对应数量
def output_correspondence_number(correspond, count, names=None, output=True):
    if names is None:
        names = []
    numbers = [0 for name in names]
    for [scene_id, number] in count:
        if correspond[scene_id] == '' and number and output:
            print(scene_id, '没分配人员')
        if correspond[scene_id] not in names:
            names.append(correspond[scene_id])
            numbers.append(0)
        id = names.index(correspond[scene_id])
        numbers[id] += number
    for i, name in enumerate(names):
        if output:
            print("{}\t{}".format(name, numbers[i]))
    return names, numbers


OUTPUT_FILE = os.path.join("./data/", "ScanVQA_train.json")
OUTPUT_FILE_FILTERED = os.path.join("./data/", "ScanVQA_train.json")
# org1, count1 = solve(".\\Files", OUTPUT_FILE)
# org2, count2 = solve(".\\Filtered_v1", OUTPUT_FILE_FILTERED, filter=True)
org2, count2 = solve(".\\Filtered_v1", OUTPUT_FILE_FILTERED, filter=False)
# correspondence, countc = get_correspondence("./ScanVQA Dataset标注示例.xlsx")
# output_by_scene(count1, 'raw')
output_by_scene(count2, 'filtered_v1')
# output_by_scene(countc, 'In the file')

# names, _ = output_correspondence_number(correspondence['标注'], count1, output=True)
# names, _ = output_correspondence_number(correspondence['筛选'], count2, names, output=True)
# names, _ = output_correspondence_number(correspondence['标注'], countc, names, output=False)
# names, _ = output_correspondence_number(correspondence['筛选'], countc, names, output=False)

# 输出哪些场景没有筛选过
# for value in [x for x in count2 if not x[1]]:
#     print(value)

# 输出哪些场景没有筛选过
print('未标注场景')
vx = [x[0] for x in count2]
for value in [x for x in range(706) if x not in vx]:
    print(value)
