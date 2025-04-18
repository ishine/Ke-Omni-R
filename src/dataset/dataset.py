import json
import logging

import torchaudio
from torch.utils.data import Dataset


def _handle_wav(wav_path, target_rate=16000):
    """
    handle one wav file.
    Return:
        waveform: numpy narray(1d)
    """
    waveform, sample_rate = torchaudio.load(wav_path)
    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_rate)(waveform)
    audio = waveform[0]
    return audio

def _handle_avqa(obj_avqa, is_think=False, think_max_len=50):
    choice_str = f"Please choose the answer from the following options: {obj_avqa['multi_choice']}."
    if is_think:
        question_template = f"{obj_avqa['question_text'].replace('video', 'audio')} {choice_str} Output the thinking process (less than {think_max_len} words) in <think> </think> and final answer in <answer> </answer>."
    else:
        question_template = f"{obj_avqa['question_text'].replace('video', 'audio')} {choice_str} Directly output the final answer in <answer> </answer>."
    #obj_avqa["prompt"] = [{"role": "system", "content": [{"type":"text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}]},
    #                      {"role": "user", "content": [{"type": "audio", "audio": obj_avqa["audio_path"]}, {"type": "text", "text": question_template}]}]
    obj_avqa["prompt"] = [{"role": "user", "content": [{"type": "audio", "audio": obj_avqa["audio_path"]}, {"type": "text", "text": question_template}]}]
    answer_str = obj_avqa["multi_choice"][obj_avqa["answer"]]
    obj_avqa["solution"] = f"<answer>{answer_str}</answer>"
    return obj_avqa

def handle_json_line(json_line, sample_rate=16000, is_think=False, think_max_len=50, load_audio=False):
    obj = json.loads(json_line)
    if load_audio:
        waveform = _handle_wav(obj["audio"], sample_rate)
        obj["audio"] = waveform.numpy()

    if obj["dataset_name"] == "AVQA":
        return _handle_avqa(obj, is_think)
    
    return obj

class AudioDataset(Dataset):
    def __init__(self, data_file, sample_rate=16000, is_perturb=False, is_think=False, think_max_len=50):
        super().__init__()
        self.lists = []
        with open(data_file, 'r', encoding='utf8') as fin:
            for line in fin:
                self.lists.append(line)

        self.sample_rate = sample_rate
        self.is_perturb = is_perturb
        self.is_think = is_think
        self.think_max_len = think_max_len
        logging.info(f"{data_file}, len:{len(self.lists)}, rate:{sample_rate}")

    def __len__(self):
        return len(self.lists)

    def __getitem__(self, index):
        return handle_json_line(self.lists[index], self.sample_rate, self.is_think, self.think_max_len)
