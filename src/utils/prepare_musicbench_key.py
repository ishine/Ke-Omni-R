import json
import os
import random
import argparse

parser = argparse.ArgumentParser(description='Prepare MusicBench dataset.')
parser.add_argument('--input_file', help="input json file", required=True)
parser.add_argument('--out_file', help="output format file", required=True)

args = parser.parse_args()
input_file = args.input_file
out_file = args.out_file

# 定义调号列表
notes = ['C', 'D', 'Db', 'E', 'Eb', 'F', 'F#', 'G', 'Gb', 'A', 'Ab', 'Bb', 'B']

equals = {'B#': 'C', 'C#': 'Db', 'D#': 'Eb', 'Fb': 'E', 'E#': 'F', 'G#': 'Ab', 'A#': 'Bb', 'Cb': 'B'}
# 生成所有可能的调式（major和minor）
all_major = [f"{note} major" for note in notes]
all_minor = [f"{note} minor" for note in notes]
all_keys = {'major': all_major, 'minor': all_minor}
    
def parse_jsonl(input_file, output_file):

    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            data = json.loads(line.strip())
            location = data['location']
            id = 'key_' +  os.path.splitext(os.path.basename(location))[0]
            key_0, key_1 = data['key']
            if key_0 in equals:
                key_0 = equals[key_0]
            correct_key = f"{key_0} {key_1}"
            audio_path = data['audio_path']
            
            # 确定调式类型（major/minor）
            key_type = key_1.lower()
            
            # 获取同类型的所有调式并排除正确答案
            candidate_keys = all_keys[key_type].copy()
            #if correct_key not in candidate_keys:
            #    continue
            candidate_keys.remove(correct_key)
            
            # 随机选择3个错误选项
            incorrect_keys = random.sample(candidate_keys, 3)
            
            # 组合选项并打乱顺序
            multi_choice = [correct_key] + incorrect_keys
            random.shuffle(multi_choice)
            
            # 确定正确答案索引
            answer = multi_choice.index(correct_key)
            
            # 构建输出JSON
            output_data = {
                'id': id,
                'question_text': "Which of the following key best fits the rhythm of this piece?",
                'multi_choice': multi_choice,
                'answer': answer,
                'audio_path': audio_path,
                'dataset_name': 'MusicBench'
            }
            
            # 写入输出文件
            outfile.write(json.dumps(output_data) + '\n')

# 使用示例
parse_jsonl(input_file, out_file)