import os
import logging

from utils.other_utils import set_env
from utils.data_utils import tokenize_dataset
from utils.model_utils import get_model, get_tokenizer_and_processor

from datasets import load_from_disk
from dataclasses import field, dataclass
from transformers import HfArgumentParser, TrainingArguments, Trainer

logger = logging.getLogger()

@dataclass
class FIMSETrainingArguments(TrainingArguments):

    # data
    max_len: int = field(default=16384)
    fim_rate: float = field(default=0.9)
    num_workers: int = field(default=64)

    train_file: str = field(default=None)

    # model
    model_cfg: str = field(default=None)
    tokenizer_cfg: str = field(default=None)

    # cache
    load_cache: bool = field(default=False)
    save_cache: bool = field(default=False)

def train():
    parser = HfArgumentParser(FIMSETrainingArguments)
    
    args = parser.parse_args_into_dataclasses()[0]
    
    set_env(args)

    if args.save_cache:
        tokenizer, processor = get_tokenizer_and_processor(args)
    else:
        model, tokenizer, processor = get_model(args)

    if args.load_cache and os.path.exists(args.train_file):
        logger.info("Loading cached dataset from %s", args.train_file)
        train_sets = load_from_disk(args.train_file)
    else:
        train_sets = tokenize_dataset(args, processor, tokenizer, args.train_file)
        if args.save_cache and args.process_index == 0:
            cache_dir = os.path.join(args.output_dir, 'cache')
            logger.info("Saving cached dataset to %s", cache_dir)
            train_sets.save_to_disk(cache_dir)

    if not args.save_cache:
        trainer = Trainer(
            args=args,
            model=model, 
            tokenizer=tokenizer,
            train_dataset=train_sets
        )

        trainer.train()

if __name__ == "__main__":

    train()
