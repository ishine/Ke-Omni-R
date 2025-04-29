import json
import os
import random
import re
import argparse

parser = argparse.ArgumentParser(description='Prepare MusicBench dataset.')
parser.add_argument('--input_file', help="input json file", required=True)
parser.add_argument('--out_file', help="output format file", required=True)

args = parser.parse_args()
input_file = args.input_file
out_file = args.out_file

def parse_jsonl(input_file, output_file):
    # 定义常见拍号列表（按使用频率排序）
    common_time_signatures = ["4/4", "3/4", "2/4", "6/8", "12/8", "5/4", "7/8", "9/8"]
    
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            data = json.loads(line.strip())
            location = data['location']
            id = 'time_' + os.path.splitext(os.path.basename(location))[0]
            
            # 提取拍号信息
            prompt_bt = data['prompt_bt']
            match = re.search(r'(\d+/\d+)', prompt_bt)
            if not match:
                continue  # 跳过无效格式
            correct_ts = match.group(1)
            
            # 特殊处理3/4和6/8互斥
            if correct_ts in ["3/4", "6/8"]:
                exclude = {"3/4", "6/8"}
                incorrect_pool = [ts for ts in common_time_signatures if ts not in exclude]
            else:
                incorrect_pool = [ts for ts in common_time_signatures if ts != correct_ts]
            
            # 生成选项（1正确 + 3错误）
            selected_incorrect = random.sample(incorrect_pool, 3)
            multi_choice = [correct_ts] + selected_incorrect
            random.shuffle(multi_choice)
            
            # 构建输出对象
            output_data = {
                'id': id,
                'question_text': "Which of the following time signatures best fits the rhythm of this piece?",
                'multi_choice': multi_choice,
                'answer': multi_choice.index(correct_ts),
                'audio_path': data['audio_path'],
                'dataset_name': 'MusicBench'
            }
            
            # 写入输出文件
            outfile.write(json.dumps(output_data) + '\n')

# 使用示例
parse_jsonl(input_file, out_file)