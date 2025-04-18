import logging
from dataclasses import dataclass, field
from typing import Optional

import transformers
from transformers import HfArgumentParser
from trl import GRPOConfig

from trainer.grpo_trainer import GRPOTrainer
from utils.rewards import accuracy_reward, format_reward
from dataset.dataset import AudioDataset


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    config_path: Optional[str] = field(default=None, metadata={"help": "config path"})
    model_name_or_path : Optional[str] = field(default=None, metadata={"help": "model name or path"})
    out_dir: Optional[str] = field(default=None, metadata={"help": "output dir for model"})
    data_file: Optional[str] = field(default=None, metadata={"help": "train data file"})
    use_wandb: Optional[str] = field(default="false", metadata={"help": "whether use wandb to report logs"})
    think: Optional[bool] = field(default=False, metadata={"help": "whether think step by step"})
    think_max_len: Optional[int] = field(
        default=50, metadata={"help": "Max length of think process"}
    )
    beta: Optional[float] = field(
        default=0.01, metadata={"help": "Beta coefficient for reward scaling in GRPO"}
    )
    num_generations: Optional[int] = field(
        default=8, metadata={"help": "Number of candidate generations per prompt"}
    )
    def __post_init__(self):
        if self.config_path is None:
            raise ValueError("config path should not none")


def main():
    parser = HfArgumentParser(DataTrainingArguments)
    data_args = parser.parse_args_into_dataclasses()[0]
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    transformers.logging.set_verbosity_info()
    logging.info(data_args)

    reward_funcs_registry = {"accuracy": accuracy_reward, "format": format_reward}
    reward_funcs = [reward_funcs_registry["accuracy"], reward_funcs_registry["format"]]

    train_dataset = AudioDataset(data_args.data_file, is_think=data_args.think, think_max_len=data_args.think_max_len)

    training_args = GRPOConfig(
        seed=42,
        data_seed=42,
        output_dir=data_args.out_dir, 
        deepspeed=data_args.config_path, 
        max_prompt_length=512,
        max_completion_length=64,
        per_device_train_batch_size=1, 
        gradient_accumulation_steps=1, 
        logging_steps=10, 
        bf16=True,
        report_to="wandb" if data_args.use_wandb == "true" else ['tensorboard'],
        gradient_checkpointing=False, 
        num_train_epochs=2,
        max_steps=6400,
        run_name="AQA-GRPO", 
        save_steps=100,
        save_only_model=True, 
        temperature=1.0,
        beta=data_args.beta,
        reward_weights=[2.0, 1.0],
        num_generations=data_args.num_generations)
    
    trainer = GRPOTrainer(
        model=data_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        think=data_args.think,
        train_dataset=train_dataset,
        eval_dataset=None)

    trainer.train()
    trainer.save_model(data_args.out_dir)


if __name__ == "__main__":
    main()
