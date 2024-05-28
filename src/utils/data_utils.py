import logging
from datasets import load_dataset

logger = logging.getLogger()


def _group_text(examples, pad_token_id, max_len):

    input_ids, labels = [], []
    final_input_ids, final_labels = [], []
    
    for _input_ids, _labels in zip(examples['input_ids'], examples['labels']):

        if len(input_ids) + len(_input_ids) > max_len:
            pad_num = max_len - len(input_ids)
            final_input_ids.append(input_ids + [pad_token_id] * pad_num)
            final_labels.append(labels + [-100] * pad_num)

            input_ids, labels = [], []
        
        if len(_input_ids) != len(_labels):
            raise ValueError(f"Input and label have mismatched lengths: {_input_ids} and {_labels}")
            
        input_ids.extend(_input_ids)
        labels.extend(_labels)
    
    if len(input_ids) > 0:
        pad_num = max_len - len(input_ids)
        final_input_ids.append(input_ids + [pad_token_id] * pad_num)
        final_labels.append(labels + [-100] * pad_num)

    return {
        "input_ids": final_input_ids,
        "labels": final_labels
    }

def tokenize_dataset(args, processor, tokenizer, file):

    logger.info('Loading %s', file)

    with args.main_process_first(desc="dataset map tokenization"):

        dataset = load_dataset('json', data_files=file, split='train')

        logger.info('Total %d case before filter', len(dataset))

        dataset = dataset.map(
            processor.process_tokenize,
            batched=True,
            num_proc=args.num_workers,
            remove_columns=list(dataset.features),
            desc="Running tokenizer on dataset",
        )
    
        logger.info('Total %d case after filter', len(dataset))

    dataset = dataset.shuffle(seed=args.seed)
    with args.main_process_first(desc="dataset map grouping"):
        dataset = dataset.map(
            _group_text,
            fn_kwargs=dict(pad_token_id=tokenizer.pad_token_id, max_len=args.max_len),
            batched=True,
            num_proc=args.num_workers,
            remove_columns=list(dataset.features),
            desc=f"Grouping texts in chunks of {args.max_len}",
        )
    
    logger.info('Total %d case after group', len(dataset))
    
    return dataset
