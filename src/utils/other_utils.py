import os
import shutil
import logging
import transformers

import torch.distributed as dist

from datetime import datetime
from transformers import set_seed

logger = logging.getLogger()

def print_args(args, name):
    max_len = max([len(k) for k in vars(args).keys()]) + 4
    logger.info(f"******************* {name} *******************")
    for key, val in sorted(vars(args).items()):
        keystr = "{}".format(key) + (" " * (max_len - len(key)))
        logger.info("%s -->   %s", keystr, val)
    logger.info(f"******************* {name} *******************\n")
 
def set_logger(_logger, local_rank, log_file=None):
    _logger.handlers.clear()
    
    if local_rank in [-1, 0]:
        _logger.setLevel(logging.INFO)
    else:
        _logger.setLevel(logging.WARN)

    log_format = '[%(asctime)s] [Rank {} - %(levelname)s] [%(filename)s - %(lineno)d] %(message)s'.format(local_rank)
    log_format = logging.Formatter(log_format, '%Y-%m-%d %H:%M:%S')
    
    console = logging.StreamHandler()
    console.setFormatter(log_format)
    _logger.addHandler(console)
    
    if log_file is not None:

        file = logging.FileHandler(log_file, mode='a')
        file.setFormatter(log_format)
        _logger.addHandler(file)

def set_env(args):
    args._frozen = False
    
    if args.tokenizer_cfg is None:
        args.tokenizer_cfg = args.model_cfg

    time = datetime.now()
    timestr = time.strftime("-%m-%d-%H:%M")

    if os.path.exists(args.output_dir):
        if args.overwrite_output_dir:
            if args.process_index == 0:
                shutil.rmtree(args.output_dir)
        else:
            raise ValueError(f"Output directory ({args.output_dir}) already exists. Use --overwrite_output_dir to overcome.")
            
    log_path = os.path.join(args.output_dir, f'log{timestr}')
    args.logging_dir = os.path.join(args.output_dir, 'logging')
    
    if args.world_size > 1:
        dist.barrier()
    
    if args.process_index == 0:
        os.makedirs(log_path, exist_ok=True)
    
    if args.world_size > 1:
        dist.barrier()
    
    node_rank = int(os.getenv('GROUP_RANK', '0'))
    for _logger in [logger, transformers.utils.logging.get_logger(), logging.getLogger('DeepSpeed')]:
        set_logger(_logger, args.local_rank, os.path.join(log_path, f'node-{node_rank}.log'))
    
    if args.world_size > 1:
        dist.barrier()

    logger.warning("Device: %s, rank: %s, world size: %s", args.device, args.process_index, args.world_size)

    if args.world_size > 1:
        dist.barrier()
    
    set_seed(args.seed)

    print_args(args, 'Training Arguments')