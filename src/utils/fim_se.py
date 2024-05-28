
import numpy as np
from transformers import LlamaTokenizer, LlamaTokenizerFast

class FIMSEProcessor:

    start_token = "<START>"
    end_token = "<END>"

    def __init__(self, max_lan, tokenizer, fim_rate=0.9):
        
        self.max_len = max_lan
        self.tokenizer = tokenizer

        self.fim_rate = fim_rate

        if isinstance(self.tokenizer, (LlamaTokenizer, LlamaTokenizerFast)):
            self.prefix_token = "<PRE>"
            self.middle_token = "<MID>"
            self.suffix_token = "<SUF>"
            self.end_token = "<EOT>"
        else:
            self.prefix_token = "<fim_prefix>"
            self.middle_token = "<fim_middle>"
            self.suffix_token = "<fim_suffix>"
            self.end_token = "<|endoftext|>"

        self.trans_dict = dict(
            prefix=self.prefix_token, 
            middle=self.middle_token, 
            suffix=self.suffix_token,
            end=self.end_token,
            startswith=self.start_token,
            endswith=self.end_token
        )

    def get_special_token(self):
        return [self.start_token, self.end_token]

    def _get_info(self, code, np_rng):

        splits = np_rng.choice(len(code), 2, replace=False)
        splits = sorted(splits)

        prefix = code[:splits[0]]
        middle = code[splits[0]:splits[1]]
        suffix = code[splits[1]:]

        prefix = prefix.split('\n')
        l_prefix, r_prefix = prefix[-1], '\n'.join(prefix[:-1]) + '\n'

        suffix = suffix.split('\n')
        f_suffix, r_suffix = suffix[0] + '\n', '\n'.join(suffix[1:])

        middle = l_prefix + middle + f_suffix

        rand = np_rng.rand()

        if rand < 0.25:
            l_prefix, f_suffix = "", ""
        elif rand < 0.50:
            l_prefix = ""
        elif rand < 0.75:
            f_suffix = ""
        
        return l_prefix, r_prefix, f_suffix, r_suffix, middle

    def _permute_code(self, code):

        np_rng = np.random.RandomState(seed=len(code))

        if not np_rng.binomial(1, self.fim_rate):
            return code

        l_prefix, r_prefix, f_suffix, r_suffix, middle = self._get_info(code, np_rng)

        return (
            f'{self.prefix_token}{r_prefix}'
            f'{self.suffix_token}{r_suffix}'
            f'{self.start_token}{l_prefix}'
            f'{self.end_token}{f_suffix}'
            f'{self.middle_token}{middle}'
            f'{self.end_token}'
        )
    
    def process_tokenize(self, examples):

        texts = []
        for example in examples['text']:
            texts.append(self._permute_code(example))

        input_ids = self.tokenizer(texts)['input_ids']
        
        final_input_ids = []
        for input_id in input_ids:
            if len(input_id) <= self.max_len:
                final_input_ids.append(input_id)
        
        return dict(input_ids=final_input_ids, labels=final_input_ids)
