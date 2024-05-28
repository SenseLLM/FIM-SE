import os
import json
import argparse

from copy import deepcopy
from vllm import LLM, SamplingParams
from human_eval.evaluation import evaluate_functional_correctness

os.environ["TOKENIZERS_PARALLELISM"] = "false"

METHODS = dict()

def registry(name):

    def _registry(_func):
        METHODS[name] = _func
        return _func
    
    return _registry
    
def load_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as fr:
        for line in fr.readlines():
            data.append(json.loads(line))
    return data

def save_jsonl(data, path, mode='w'):
    with open(path, mode, encoding='utf-8') as fw:
        for d in data:
            fw.write(json.dumps(d, ensure_ascii=False) + '\n')

@registry('fim')
class FIM:

    @classmethod
    def _get_prefix(cls, data):

        prefix, suffix = data['prompt'], data['suffix']

        prompt = f'{prefix_token}{prefix}{suffix_token}{suffix}{middle_token}'

        return prompt

    @classmethod
    def _update_data(cls, data, result):

        prefix, suffix = data['prompt'], data['suffix']

        data['output'] = result

        data['completion'] =  prefix + result + suffix
        data['task_id'] = '/'.join(data['task_id'].split('/')[1:-1])

        return data

@registry('fim_spm')
class FIM_SPM(FIM):

    @classmethod
    def _get_prefix(cls, data):

        prefix, suffix = data['prompt'], data['suffix']

        prompt = f'{prefix_token}{suffix_token}{suffix}{middle_token}{prefix}'

        return prompt

@registry('fim_se')
class FIM_SE:

    @classmethod
    def _get_info(cls, data):
        if data['task_id'].startswith('RandomSpanInfilling'):
            prefix, suffix = data['prompt'], data['suffix']

            prefix = prefix.split('\n')
            l_prefix, r_prefix = prefix[-1], '\n'.join(prefix[:-1]) + '\n'

            suffix = suffix.split('\n')
            f_suffix, r_suffix = suffix[0] + '\n', '\n'.join(suffix[1:])

            return l_prefix, r_prefix, f_suffix, r_suffix

        return "", data['prompt'], "", data['suffix']

    @classmethod
    def _get_prefix(cls, data):
        l_prefix, r_prefix, f_suffix, r_suffix = cls._get_info(data)

        prompt = f'{prefix_token}{r_prefix}{suffix_token}{r_suffix}<START>{l_prefix}<END>{f_suffix}{middle_token}'

        return prompt

    @classmethod
    def _update_data(cls, data, result):
        l_prefix, r_prefix, f_suffix, r_suffix = cls._get_info(data)

        data['output'], data['l_prefix'], data['f_suffix'] = result, l_prefix, f_suffix

        data['completion'] = ''
        if result.startswith(l_prefix) and result.endswith(f_suffix):
            data['completion'] = r_prefix + result + r_suffix

        data['task_id'] = '/'.join(data['task_id'].split('/')[1:-1])
        
        return data

def test_huamneval(task, method, model, generate_params):
    trans = {
        "single-line": "SingleLineInfilling",
        "multi-line": "MultiLineInfilling",
        "random-span": "RandomSpanInfilling",
    }

    os.makedirs(f'{args.model_cfg}/results', exist_ok=True)

    name = f'{task}-{method}'

    in_file = f'data/infilling/HumanEval-{trans[task]}.jsonl'
    out_file = f'{args.model_cfg}/results/{name}.jsonl'

    data = load_jsonl(in_file)

    prompts = [METHODS[method]._get_prefix(d) for d in data]
    print(prompts[0])
    
    completions = model.generate(prompts, generate_params)
    print(completions[0])
    
    outs = []
    for d, completion in zip(data, completions):
        for output in completion.outputs:
            outs.append(METHODS[method]._update_data(deepcopy(d), output.text))

    save_jsonl(outs, out_file)
    evaluate_functional_correctness(out_file)
    
    results = load_jsonl(out_file + '_results.jsonl')

    score = round(sum([r['passed'] for r in results]) / len(results), 3)
    print(score)

    scores = {}
    if os.path.exists(f'{args.model_cfg}/results/score.jsonl'):
        with open(f'{args.model_cfg}/results/score.jsonl', 'r') as fr:
            scores = json.load(fr)
    
    scores[f'{method}-{name}'] = score
    with open(f'{args.model_cfg}/results/score.jsonl', 'w') as fw:
        json.dump(scores, fw, indent=4, sort_keys=True)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_cfg', default='starcoder', type=str)
    parser.add_argument('--token_mode', default='llama', type=str)
    
    parser.add_argument('--task', default='random-span', type=str)
    parser.add_argument('--method', default='fim_se', type=str)

    args = parser.parse_args()

    if args.token_mode == 'llama':
        prefix_token, middle_token, suffix_token, end_token = "<PRE>", "<MID>", "<SUF>", "<EOT>"
    elif args.token_mode == 'starcoder':
        prefix_token, middle_token, suffix_token, end_token = "<fim_prefix>", "<fim_middle>", "<fim_suffix>", "<|endoftext|>"
    else:
        raise NotImplementedError
    
    sample_params = SamplingParams(
        temperature=0,
        max_tokens=256,
        stop=end_token,
        spaces_between_special_tokens=False
    )

    model = LLM(
        model=args.model_cfg,
        trust_remote_code=True, 
        tensor_parallel_size=1
    )

    if args.task == 'all':
        args.task = ['random-span', 'single-line', 'multi-line']
    else:
        args.task = [args.task]
        
    for task in args.task:
        test_huamneval(task, args.method, model, sample_params)
