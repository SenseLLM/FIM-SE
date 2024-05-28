from typing import List

from transformers import (
    LlamaTokenizer,
    LlamaTokenizerFast,
    SLOW_TO_FAST_CONVERTERS,
)

from tokenizers import decoders, normalizers
from transformers.convert_slow_tokenizer import LlamaConverter

SPIECE_UNDERLINE = "▁"

class FixLlamaTokenizer(LlamaTokenizer):

    def __init__(self, *args, **kwargs):
        kwargs['legacy'] = False
        super().__init__(*args, **kwargs)

    def tokenize(self, text, **kwargs) -> List[int]:
        
        return super().tokenize(text.replace(SPIECE_UNDERLINE, " "), **kwargs)

    def _tokenize(self, text, **kwargs):
        return self.sp_model.encode(text, out_type=str)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""

        current_sub_tokens = []
        out_string = ""
        for _, token in enumerate(tokens):
            # make sure that special tokens are not decoded using sentencepiece model
            if token in self.all_special_tokens:
                out_string += self.sp_model.decode(current_sub_tokens) + token
                current_sub_tokens = []
            else:
                current_sub_tokens.append(token)
        out_string += self.sp_model.decode(current_sub_tokens)
        return out_string

    def _decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
        spaces_between_special_tokens: bool = False,
        **kwargs,
    ) -> str:
        self._decode_use_source_tokenizer = kwargs.pop("use_source_tokenizer", False)

        filtered_tokens = self.convert_ids_to_tokens(token_ids, skip_special_tokens=skip_special_tokens)
        legacy_added_tokens = set(self._added_tokens_encoder.keys()) - set(self.all_special_tokens) | {
            token for token in self.additional_special_tokens if self.convert_tokens_to_ids(token) >= self.vocab_size
        }
        # To avoid mixing byte-level and unicode for byte-level BPT
        # we need to build string separately for added tokens and byte-level tokens
        # cf. https://github.com/huggingface/transformers/issues/1133
        sub_texts = []
        current_sub_text = []
        # TODO @ArthurZ in version 5, special tokens should be handled in convert_tokens_to_string, while _convert_tokens_to_string
        for token in filtered_tokens:
            if skip_special_tokens and token in self.all_special_ids:
                continue
            if token in legacy_added_tokens:
                if current_sub_text:
                    string = self.convert_tokens_to_string(current_sub_text)
                    if len(string) > 0:
                        sub_texts.append(string)
                    current_sub_text = []
                sub_texts.append(token)
            else:
                current_sub_text.append(token)
        if current_sub_text:
            sub_texts.append(self.convert_tokens_to_string(current_sub_text))

        if spaces_between_special_tokens:
            text = " ".join(sub_texts)
        else:
            text = "".join(sub_texts)

        clean_up_tokenization_spaces = (
            clean_up_tokenization_spaces
            if clean_up_tokenization_spaces is not None
            else self.clean_up_tokenization_spaces
        )
        if clean_up_tokenization_spaces:
            clean_text = self.clean_up_tokenization(text)
            return clean_text
        else:
            return text

class FixLlamaTokenizerFast(LlamaTokenizerFast):
    slow_tokenizer_class = FixLlamaTokenizer

class FixLlamaTokenizerConverter(LlamaConverter):
    def normalizer(self, proto):
        return normalizers.Replace(pattern=' ', content='▁')

    def decoder(self, replacement, add_prefix_space):
        return decoders.Sequence(
            [
                decoders.Replace('▁', ' '),
                decoders.ByteFallback(),
                decoders.Fuse(),
            ]
        )

FixLlamaTokenizer.register_for_auto_class()
FixLlamaTokenizerFast.register_for_auto_class()
SLOW_TO_FAST_CONVERTERS[FixLlamaTokenizer.__name__] = FixLlamaTokenizerConverter
