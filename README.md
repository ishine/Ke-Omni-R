# Ke-Omni-R: Achieving Advanced Audio Reasoning with a Concise 50-Words Think Process
![GitHub](https://img.shields.io/github/license/shuaijiang/Ke-Omni-R)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging_Face-Ke--Omni--R-blue.svg)](https://huggingface.co/KE-Team/Ke-Omni-R/)

## Introduction
Ke-Omni-R is an advanced audio reasoning model built upon [Qwen2.5-Omni-7B](https://github.com/QwenLM/Qwen2.5-Omni). With only 10k post-training samples, Ke-Omni-R has achieved state-of-the-art performance on the MMAU *Test-mini* and *Test* benchmarks. Key insights from its development include:

- **GRPO Algorithm**: The GRPO algorithm significantly enhances the performance of the already strong base model (Qwen2.5-Omni-7B), demonstrating superior generalization even in unseen speech domains.
- **Think Process**: Incorporating a concise think process (less than 50 words) plays a crucial role in improving reasoning capabilities.
- **KL Divergence**: Slight improvements were observed during GRPO training by leveraging KL divergence.
- **Domain Ratio vs. Data Volume**: Domain diversity outweighs data volume. We utilized only 10k samples, with 5k randomly selected from AVQA and another 5k from MusicBench.


## News
- April 17, 2025: Released model [🤗 Ke-Omni-R](https://huggingface.co/KE-Team/Ke-Omni-R)

---
## Content
- [Introduction](#introduction)
- [News](#news)
- [Performance: Accuracies (%) on MMAU Test-mini and Test benchmark](#performance-accuracies--on-mmau-test-mini-and-test-benchmark)
- [Roadmap](#roadmap)
- [Quickstart](#quickstart)
  - [Installation](#installation)
  - [First Demo](#first-demo)
- [Training (codes coming soon)](#trainingcodes-coming-soon)
  - [Training Data](#training-data)
  - [Training Strategy](#training-strategy)
- [Testing on MMAU](#testing-on-mmau)
  - [Step 1: Download Dataset](#step-1-download-dataset)
  - [Step 2: Format Data](#step-2-format-data)
  - [Step 3: Evaluation](#step-3-evaluation)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)

---

## Performance: Accuracies (%) on MMAU Test-mini and Test benchmark
| Model                                 | Method                | Sound (Test-mini) | Sound (Test)  | Music (Test-mini) | Music (Test)  | Speech (Test-mini) | Speech (Test)  | Average (Test-mini) | Average (Test)  |
|---------------------------------------|-----------------------|-----------|-------|-----------|-------|-----------|------|------------|-------|
| -                                     | Human\*               | 86.31     | -     | 78.22     | -     | 82.17     | -     | 82.23     | -     |
| Gemini Pro 2.0 Flash                  | Direct Inference\*    | 56.46     | 61.73 | 58.68     | 56.53 | 51.65     | 61.53 | 55.60     | 59.93 |
| Audio Flamingo 2                      | Direct Inference\*    | 61.56     | 65.10 | **73.95** |**72.90**| 30.93     | 40.26 | 55.48     | 59.42 |
| GPT4o + Strong Cap.                   | Direct Inference\*    | 57.35     | 55.83 | 49.70     | 51.73 | 64.86     | **68.66** | 57.30     | 58.74 |
| Llama-3-8B-Instruct + Strong Cap.     | Direct Inference\*    | 50.75     | 49.10 | 48.93     | 48.93 | 55.25     | 62.70 | 52.10     | 53.57 |
| Qwen2-Audio-7B-Instruct               | Direct Inference\*    | 54.95     | 45.90 | 50.98     | 53.26 | 42.04     | 45.90 | 49.20     | 52.50 |
| SALAMONN                              | Direct Inference\*    | 41.00     | 40.30 | 34.80     | 33.76 | 25.50     | 24.24 | 33.70     | 32.77 |
| Audio-Reasoner(Qwen2-Audio-7B-Instruct) | \[1\]               | 60.06     | -     | 64.30     | -     | 60.70     | -     | 61.71     | -     |
| Audio-Cot(Qwen2-Audio-7B-Instruct)    | \[2\]                 | 61.86     | -     | 56.29     | -     | 55.26     | -     | 57.80     | -     |
| R1-AQA(Qwen2-Audio-7B-Instruct)       | \[3\]                 | 68.77     | 69.76 | 64.37     | 61.40 | 63.66     | 62.70 | 65.60     | 64.36 |
| Qwen2.5-Omni-7B                       | \[4\]                 | 67.87     | -     | 69.16     | -     | 59.76     | -     | 65.60     | -     |
| Ke-Omni-R(Qwen2.5-Omni-7B)            | GRPO(ours)            | **69.37** | **71.90** | 69.46 | 67.13 |**67.87**  | 67.10 | **68.90** |**68.71** |

Note:
- \* The data are sourced from the [MMAU leaderboard](https://sakshi113.github.io/mmau_homepage/#leaderboard).
- \[1\] Xie, Zhifei, et al. "Audio-Reasoner: Improving Reasoning Capability in Large Audio Language Models." arXiv preprint arXiv:2503.02318.  
- \[2\] Ma, Ziyang, et al. "Audio-CoT: Exploring Chain-of-Thought Reasoning in Large Audio Language Model." arXiv preprint arXiv:2501.07246.
- \[3\] Li, Gang, et al. "Reinforcement Learning Outperforms Supervised Fine-Tuning: A Case Study on Audio Question Answering." arXiv preprint arXiv:2503.11197
- \[4\] Xu, Jin, et al. "Qwen2.5-Omni Technical Report." arXiv preprint arXiv:2503.20215

## Roadmap

- [x] 2025/04
    - [x] [Ke-Omni-R](https://huggingface.co/KE-Team/Ke-Omni-R) models released

- [ ] 2025/04
    - [ ] Training codes release
    - [ ] Training data release

## Quickstart
### Installation
The codebase for Qwen2.5-Omni is integrated into the latest Hugging Face `transformers` library. It is recommended to build from the source using the following commands:
```
pip uninstall transformers
pip install git+https://github.com/huggingface/transformers
pip install accelerate
```
### First Demo
```python
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

# You can directly insert a local file path, a URL, or a base64-encoded audio into the position where you want in the text.
messages = [
  # Audio
    ## Local audio path
    [{"role": "system", "content":[{"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}]},
     {"role": "user", "content": [{"type": "audio", "audio": "/path_to_avqa_wavs/-IBtBeR6B00_000000.wav"}, {"type": "text", "text": "Please describe this audio."}]}],
    [{"role": "user", "content": [{"type": "audio", "audio": "/path_to_avqa_wavs/-IBtBeR6B00_000000.wav"}, {"type": "text", "text": "What is the main source of sound in the audio? ['aircraft', 'Car', 'Tank', 'Missile'] Output the thinking process (less than 50 words) in <think> </think> and final answer in <answer> </answer>."}]}],
    [{"role": "user", "content": [{"type": "audio", "audio": "/path_to_avqa_wavs/-IBXTktoom8_000030.wav"}, {"type": "text", "text": "What animal is the main source of sound in the video? ['dog', 'wasp', 'honeybee', 'dragonfly'] Output the thinking process (less than 50 words) in <think> </think> and final answer in <answer> </answer>."}]}],
]

model = Qwen2_5OmniForConditionalGeneration.from_pretrained('KE-Team/Ke-Omni-R')
processor = Qwen2_5OmniProcessor.from_pretrained(model_path)

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(text)
audios, images, videos = process_mm_info(messages, use_audio_in_video=False)
inputs = processor(text=text, images=images, videos=videos, audio=audios, padding=True, return_tensors="pt")

generation = model.generate(**inputs, thinker_temperature=0, thinker_do_sample=False)
generated_ids = generation[:, inputs.input_ids.size(1):]
completions = processor.batch_decode(generated_ids, skip_special_tokens=True)
print(completions)
```

the output should be
```
["Well, it sounds like there's a car accelerating. You can hear the engine revving up, and there's a bit of a thump or thud sound too. It might be the car hitting something or just a part of the acceleration process. It gives off a sense of speed and power. What do you think about it? Do you have any other audio samples you want to talk about?", '<think>The audio features a vehicle accelerating and revving, which is characteristic of a car. The sound is consistent with a car engine, not an aircraft, tank, or missile.</think>\n<answer>Car</answer>', "<think>The main source of sound is a buzzing insect, which is consistent with the size and sound of a honeybee. The other options don't match the sound or context.</think>\n<answer>honeybee</answer>"]
```

---
## Training(codes coming soon)
### Training Data
Ke-Omni-R was trained on two datasets:
- **AVQA**: 5k randomly selected samples.
- **MusicBench**: 5k randomly selected samples.

### Training Strategy
- **Weighted GRPO Training**: Accuracy and format reward functions are weighted at a ratio of 2:1.
- **Think Process**: A concise think process (less than 50 words) was included during training. The output format is as follows:
  ```
  <think> Thinking process (less than 50 words) </think>
  <answer> Final answer </answer>
  ```
- **KL Divergence**: Applied during GRPO training to slightly improve performance.


## Testing on MMAU
To evaluate the model on the MMAU Test-mini dataset, follow these steps:
### Step 1: Download Dataset
 Download the MMAU Test-mini dataset: [test-mini-audios.tar.gz](https://drive.google.com/file/d/1fERNIyTa0HWry6iIG1X-1ACPlUlhlRWA/view?usp=sharing) 

```bash
mkdir -p data && cd data

git clone https://github.com/Sakshi113/MMAU.git

cd MMAU

#TODO you should download test-mini-audios.tar.gz to here
***download test-mini-audios.tar.gz to here***

# Uncompress wav files
tar -xzvf test-mini-audios.tar.gz

cd ../../
```

### Step 2: Format Data
```bash
python src/utils/prepare_mmau.py \
    --input_file data/MMAU/mmau-test-mini.json \
    --wav_dir data/MMAU/test-mini-audios \
    --out_file data/MMAU/mmau-mini.data
```

### Step 3: Evaluation
Run the evaluation script:
```bash
# Test MMAU test-mini every 100 steps. Modify the script to test other steps or parameters if needed.
sh test_mmau.sh
```
---
## Acknowledgements
We express our gratitude to the following projects and teams for their contributions:
- **R1-AQA**: Referenced the GRPO-based training implementation from [R1-AQA](https://github.com/xiaomi-research/r1-aqa).
- **Qwen Team**: Special thanks to the [Qwen2.5-Omni-7B](https://github.com/QwenLM/Qwen2.5-Omni) model for providing a robust foundation.
- **Datasets**: 
  - [AVAQ](https://mn.cs.tsinghua.edu.cn/avqa/)
  - [MusicBench](https://amaai-lab.github.io/mustango/)
  - [MMAU](https://github.com/Sakshi113/MMAU/)


## Citation
```bib
@misc{zhao2025keomnir,
  author = {Zhao, Shuaijiang and Guo, Tingwei and Wen, Cheng and Xiang, Bajian and Zou, Wei},
  title = {Ke-Omni-R: Achieving Advanced Audio Reasoning with a Concise 50-Words Think Process},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/shuaijiang/Ke-Omni-R}},
}
```
