"""Score selected existing responses with the fixed reward model; no generation."""
import gc
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
from common.data import load_yaml, read_jsonl, repo_path, prompt_messages, prompt_messages_from_preference
from common.models import load_reward_model
from common.generation import score_reward_pairs
from common.logging_utils import append_jsonl, save_json
from common.metrics import word_count


def main():
    if not torch.backends.mps.is_available():
        raise RuntimeError("Run outside the sandbox with MPS available for timely scoring")
    path = repo_path("results/completion_audit/scored_final_policy_examples.jsonl")
    existing = read_jsonl(path) if path.exists() else []
    done = {(r['task'],r['run'],r['prompt_id']) for r in existing}
    t1_ids = ['e64ed52b2b74c3b8fc2d9e135c7b9c5d7975282d892d5e11ff27ee2bc40cfdbf',
        'd308da7d7d4348f3021d791f8a1d234ed7c586f74e83b8feebd433073206e494',
        '816d427fd024859df1a5eb44dcb52bbe7ab01d3324d57e115e60288cbe2ff04e',
        '5a69a1aaf348618d5552e167819ed8ac9918191f68981c1c479d2d23f35aef26']
    t2_ids = ['d31b039922e3bb89d63a8f7331c90d77437a449e4b5f757a114b28002ffae60a',
        '4fa4a88327b6c9c9b36eaeb14ab4c57edb9c414319b4e0eeb237154eccd38177',
        'eec33c11c0aaf62414ade529ece498b6ecb73d874a032f0c7858751d6981e800',
        'cee55c63851c6cb6dfc6707cadd7aaf15176fff704181aa1c2123f1327bc471d',
        '4757a77e3611659864043267aea61c38ebab2f02773047426aecac4307ede8a4']
    task1 = {r['prompt_id']:r for r in read_jsonl('data/dpo_length_stratified_eval.jsonl')}
    task2 = {r['prompt_id']:r for r in read_jsonl('data/rl_prompt_pool_eval.jsonl')}
    candidates=[]
    for task, ids, source, runs in [
        ('task1_dpo',t1_ids,task1,['standard_length_stratified','length_balanced_stratified']),
        ('task2_ppo',t2_ids,task2,['standard','kl_0_00','kl_0_10','kl_0_20'])]:
        for run in runs:
            filename=f'results/{task}/{run}/generations.jsonl'
            generations={r.get('id',r.get('prompt_id')):r for r in read_jsonl(filename)}
            for pid in ids:
                if (task,run,pid) in done: continue
                r=source[pid]; gen=generations[pid]
                messages=prompt_messages_from_preference(r) if task=='task1_dpo' else prompt_messages(r)
                candidates.append((task,run,pid,filename,r,messages,gen))
    if not candidates:
        print('All selected existing responses already scored')
        return
    cfg=load_yaml('configs/ppo.yaml');cfg['reward_device']='mps'
    model,tok=load_reward_model(cfg)
    for i,(task,run,pid,filename,row,messages,gen) in enumerate(candidates):
        max_length=1024 if task=='task1_dpo' else 1280
        with torch.no_grad():
            scores=score_reward_pairs(model,tok,[messages],[gen['response']],max_length=max_length)
        item={'task':task,'run':run,'prompt_id':pid,'source_file':filename,
            'prompt_messages':messages,'response':gen['response'],'response_tokens':gen['response_length'],
            'word_count':word_count(gen['response']),'characters':len(gen['response']),
            'reward_model_score':float(scores[0]),'reward_max_length':max_length,
            'length_stratum':row.get('length_stratum'),
            'score_provenance':'New post-hoc fixed-reward scoring of original saved final-policy text; not an original per-example score. No generation or training.'}
        if not torch.isfinite(scores).all():raise ValueError('Nonfinite reward score')
        append_jsonl(path,item);existing.append(item)
        print(f'Scored saved response {i+1}/{len(candidates)}: {task}/{run} {pid[:10]}',flush=True)
    save_json('results/completion_audit/scored_final_policy_examples_summary.json',
        {'examples':len(existing),'reward_model':cfg['reward_model'],'source':str(path),
         'selection':'Four Task 1 prompts and five PPO prompts selected for directly inspectable instruction-following differences; illustrative, not a random quality estimate.'})


if __name__=='__main__':main()
