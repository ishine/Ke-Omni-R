import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Optional

import torch
import transformers
from tqdm import tqdm
from transformers import HfArgumentParser
from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

@dataclass
class TestArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    model_path: Optional[str] = field(default=None, metadata={"help": "model dir"})
    out_file: Optional[str] = field(default=None, metadata={"help": "output file for test"})
    data_file: Optional[str] = field(default=None, metadata={"help": "test file"})
    force: Optional[bool] = field(default=False, metadata={"help": "force test"})
    batch_size: Optional[int] = field(default=16, metadata={"help": "Batch size for processing"})
    think: Optional[bool] = field(default=False, metadata={"help": "whether think step by step"})
    think_max_len: Optional[int] = field(
        default=50, metadata={"help": "Max length of think process, 0 for unlimit"}
    )
    
    def __post_init__(self):
        if self.model_path is None:
            raise ValueError("config path should not none")


def _get_message(obj_dict, think=False, think_max_len=50):
    choice_str = f"Please choose the answer from the following options: {obj_dict['choices']}."
    if think:
        if think_max_len <= 0:
            question_template = f"{obj_dict['question']} {choice_str} Output the thinking process in <think> </think> and final answer in <answer> </answer>."
        else:
            question_template = f"{obj_dict['question']} {choice_str} Output the thinking process(less than {think_max_len} words) in <think> </think> and final answer in <answer> </answer>."
    else:
        question_template = f"{obj_dict['question']} {choice_str} Directly output the final answer in <answer> </answer>."    
    print(question_template)
    message = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": f"{obj_dict['audio_id']}"},
                {"type": "text", "text": question_template},
            ],
        }
    ]
    return message


def main():
    parser = HfArgumentParser(TestArguments)
    data_args = parser.parse_args_into_dataclasses()[0]
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    transformers.logging.set_verbosity_info()
    logging.info(data_args)

    if not data_args.force and os.path.exists(data_args.out_file) and os.path.getsize(data_args.out_file) > 0:
        logging.info(f"The {data_args.out_file} exists. Do not regenerate it.")
        return

    qwen2_omni_processor = Qwen2_5OmniProcessor.from_pretrained(data_args.model_path)
    qwen2_omni_model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        data_args.model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )

    datas = []
    with open(data_args.data_file, "r", encoding="utf8") as reader:
        datas = json.load(reader)

    all_outputs = []
    batch_size = data_args.batch_size
    for i in tqdm(range(0, len(datas), batch_size)):
        batch_data = datas[i : i + batch_size]

        batch_messages = []
        batch_audios = []
        for bd in batch_data:
            batch_messages.append(_get_message(bd, data_args.think, data_args.think_max_len))
        
        batch_audios, _, _ = process_mm_info(batch_messages, use_audio_in_video=False)

        text = qwen2_omni_processor.apply_chat_template(batch_messages, add_generation_prompt=True, tokenize=False)
        inputs = qwen2_omni_processor(
            text=text, audio=batch_audios, sampling_rate=16000, return_tensors="pt", padding=True
        ).to(qwen2_omni_model.device).to(torch.bfloat16)

        generated_ids = qwen2_omni_model.generate(**inputs, max_new_tokens=256)
        generated_ids = generated_ids[:, inputs.input_ids.size(1) :]
        response = qwen2_omni_processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        all_outputs.extend(response)
        print(f"Processed batch {i//batch_size + 1}/{(len(datas) + batch_size - 1)//batch_size}")

    def extract_answer(output_str):
        answer_pattern = r"<answer>(.*?)</answer>"
        match = re.search(answer_pattern, output_str)

        if match:
            return match.group(1)
        return output_str

    final_output = []
    for input_example, model_output in zip(datas, all_outputs):
        original_output = model_output
        model_answer = extract_answer(original_output).strip()

        # Create a result dictionary for this example
        result = input_example
        result["model_output"] = model_answer
        result["model_output_ori"] = original_output
        final_output.append(result)

    # Save results to a JSON file
    output_path = data_args.out_file
    with open(output_path, "w") as f:
        json.dump(final_output, f, indent=2)

    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
