import re
import os
import json
import random
import argparse

parser = argparse.ArgumentParser(description='Prepare MusicBench dataset.')
parser.add_argument('--input_file', help="input json file", required=True)
parser.add_argument('--out_file', help="output format file", required=True)
#parser.add_argument('--wav_dir', help="wav dir", required=True)

args = parser.parse_args()
input_file = args.input_file
out_file = args.out_file
#wav_dir = args.wav_dir


# 定义意大利术语及其BPM范围，按顺序排列以确保正确匹配
terms = [
    ('larghissimo', 0, 25),
    ('grave', 25, 45),
    ('largo', 40, 60),
    ('lento', 45, 60),
    ('larghetto', 60, 66),
    ('adagio', 66, 76),
    ('adagietto', 72, 76),
    ('andante', 76, 108),
    ('andantino', 80, 108),
    ('andante moderato', 92, 112),
    ('moderato', 108, 120),
    ('allegretto', 112, 120),
    ('allegro moderato', 116, 120),
    ('allegro', 120, 168),
    ('vivace', 140, 176),
    ('vivacissimo', 172, 176),
    ('allegrissimo', 172, 176),
    ('presto', 168, 200),
    ('prestissimo', 200, float('inf'))
]
all_terms = [term for term, _, _ in terms]

def handle_musicbench(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            data = json.loads(line.strip())
            location = data['location']
            id = 'tempo_' + os.path.splitext(os.path.basename(location))[0]
            bpm = data['bpm']
            if bpm is None:
                continue
            #bpm = float(re.search(r'\d+\.\d+', bpm_str).group())
            audio_path = data['audio_path']
            
            # 查找正确的术语
            correct_term = None
            for term_info in terms:
                term, min_bpm, max_bpm = term_info
                if min_bpm <= bpm <= max_bpm:
                    correct_term = term
                    break
            
            # 收集所有术语并排除正确选项
            incorrect_terms = [term for term in all_terms if term != correct_term]
            
            # 随机选择三个错误选项
            selected_incorrect = random.sample(incorrect_terms, 3)
            multi_choice = [correct_term] + selected_incorrect
            random.shuffle(multi_choice)
            
            # 确定正确答案的索引
            answer = multi_choice.index(correct_term)
            
            # 构建输出JSON对象
            output_data = {
                'id': id,
                'question_text': "Which of the following tempo mark best fits the rhythm of this piece?",
                'multi_choice': multi_choice,
                'answer': answer,
                'audio_path': audio_path,
                'dataset_name': 'MusicBench'
            }
            
            # 写入输出文件
            outfile.write(json.dumps(output_data) + '\n')
            

#handle_musicbench(input_file, out_file)


# 定义意大利术语及其BPM范围，按优先级顺序排列
TERMS = [
    ('larghissimo', 0, 25),
    ('grave', 25, 45),
    ('largo', 40, 60),
    ('lento', 45, 60),
    ('larghetto', 60, 66),
    ('adagio', 66, 76),
    ('adagietto', 72, 76),
    ('andante', 76, 108),
    ('andantino', 80, 108),
    ('andante moderato', 92, 112),
    ('moderato', 108, 120),
    ('allegretto', 112, 120),
    ('allegro moderato', 116, 120),
    ('allegro', 120, 168),
    ('vivace', 140, 176),
    ('vivacissimo', 172, 176),
    ('allegrissimo', 172, 176),
    ('presto', 168, 200),
    ('prestissimo', 200, float('inf'))
]

# 定义意大利术语及其BPM范围，使用正确的大小写形式
TERMS = [
    ('Larghissimo', 0, 25),
    ('Grave', 25, 45),
    ('Largo', 40, 60),
    ('Lento', 45, 60),
    ('Larghetto', 60, 66),
    ('Adagio', 66, 76),
    ('Adagietto', 72, 76),
    ('Andante', 76, 108),
    ('Andantino', 80, 108),
    ('Andante moderato', 92, 112),
    ('Moderato', 108, 120),
    ('Allegretto', 112, 120),
    ('Allegro moderato', 116, 120),
    ('Allegro', 120, 168),
    ('Vivace', 140, 176),
    ('Vivacissimo', 172, 176),
    ('Allegrissimo', 172, 176),
    ('Presto', 168, 200),
    ('Prestissimo', 200, float('inf'))
]
# 提取术语名称
terms_list = [term[0] for term in TERMS]

def parse_jsonl(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            data = json.loads(line.strip())
            location = data['location']
            id = 'tempo_' + os.path.splitext(os.path.basename(location))[0]
            prompt_bpm = data['prompt_bpm']
            
            pattern = r'\b(' + '|'.join(map(re.escape, terms_list)) + r')\b'
            match = re.search(pattern, prompt_bpm, re.IGNORECASE)
            if not match:
                continue  # 跳过无法解析术语的条目
            
            correct_term = match.group(1)
            # 查找对应的BPM区间
            correct_min, correct_max = 0, 0
            for term_info in TERMS:
                if term_info[0].lower() == correct_term.lower():
                    correct_min, correct_max = term_info[1], term_info[2]
                    break
            
            # 收集不重叠的候选术语
            candidate_terms = []
            for term_info in TERMS:
                term, min_t, max_t = term_info
                # 检查区间是否与正确答案的区间重叠（包括端点）
                if not (min_t <= correct_max and max_t >= correct_min):
                    candidate_terms.append(term)
            
            # 排除正确术语并随机选择3个错误选项
            incorrect_terms = [t for t in candidate_terms if t != correct_term]
            if len(incorrect_terms) < 3:
                continue  # 跳过候选不足的情况
            selected_incorrect = random.sample(incorrect_terms, 3)
            
            # 组合选项并打乱顺序
            multi_choice = [correct_term] + selected_incorrect
            random.shuffle(multi_choice)
            
            # 构建输出数据
            output_data = {
                'id': id,
                'question_text': "Which of the following tempo mark best fits the rhythm of this piece?",
                'multi_choice': multi_choice,
                'answer': multi_choice.index(correct_term),
                'audio_path': data['audio_path'],
                'dataset_name': 'MusicBench'
            }
            
            # 写入输出文件
            outfile.write(json.dumps(output_data) + '\n')

# 使用示例
parse_jsonl(input_file, out_file)