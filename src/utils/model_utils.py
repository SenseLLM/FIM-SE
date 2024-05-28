import torch

from utils.fim_se import FIMSEProcessor
from transformers.tokenization_utils import AddedToken
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, GPTBigCodeConfig

def get_tokenizer_and_processor(args):

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_cfg, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    processor = FIMSEProcessor(args.max_len, tokenizer, args.fim_rate)
    
    special_tokens = [AddedToken(t) for t in processor.get_special_token()]
    
    tokenizer.add_special_tokens(
        dict(additional_special_tokens=special_tokens), 
        replace_additional_special_tokens=False
    )

    return tokenizer, processor

def get_model(args):

    config = AutoConfig.from_pretrained(args.model_cfg)

    if isinstance(config, GPTBigCodeConfig):
        config.attention_softmax_in_fp32 = False
        config.scale_attention_softmax_in_fp32 = False
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_cfg, 
        config=config,
        torch_dtype=torch.bfloat16, 
        attn_implementation="flash_attention_2"
    )

    tokenizer, processor = get_tokenizer_and_processor(args)

    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer, processor