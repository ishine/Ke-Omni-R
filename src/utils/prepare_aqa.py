import json
import os
import re
import argparse
from tqdm import tqdm

'''# 输入数据示例
 [
 {'id': 184, 'video_name': '-HIPq7T3eFI_000011', 'video_id': 342, 'question_text': 'What is the main source of sound in the video?', 'multi_choice': ['Car', 'motorcycle', 'siren', 'gun fire'], 'answer': 0, 'question_relation': 'Both', 'question_type': 'Come From'}
 ]
'''

def create_data(input_file, audio_path, output_file):
    with open(input_file, "r") as f, open(output_file, "w") as fout:
        samples = json.load(f)
        for sample in tqdm(samples):
            if sample['question_relation'] == 'View': # skip the sample only need view 
                continue
            if re.search('color', sample['question_text']): # skip the sample related to color
                continue
            video_name = sample['video_name']
            audio_name = video_name + '.wav'
            audio_path = os.path.join(audio_path, audio_name)
            if not os.path.exists(audio_path):
                print(f"Warning: {audio_path} do not exist!")
                continue
            sample['audio_path'] = audio_path
            sample['dataset_name'] = 'AVQA'
            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()
    create_data(args.input_file, args.audio_path, args.output_file)
