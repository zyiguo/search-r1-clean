"""Cloud-only Search-R1-style SFT / fresh-rollout GRPO for one GPU.

JSON actions replace think/search tags. Bounded question groups per sampling round.
Every action is conditioned on its exact generation prefix; observations get no loss.
"""
import argparse
import json
import random
import time
from contextlib import contextmanager
from pathlib import Path
from common import ROOT, path, read_json, write_json, sha256, write_json_gzip
from grpo_sampling import collect_groups, sampling_options, training_work, reward_weights, QuestionSampler
from search_task import (Retriever, advantages, encode_step, load_bundle, reward, parallel_rollouts,
                         rollout, sft_steps, batched_rollouts, cross_question_rollouts)
from training_common import (CompletionCollator, compute_dtype, load_model, load_tokenizer,
                            lora_config, model_directory, require_cloud)


def identity(c, data_identity, merged):
    result = {'config': c, 'data': data_identity,
            'code': {p.name: sha256(p) for p in ROOT.glob('*.py')},
            'requirements': sha256('requirements-train.txt'),
            'model_origin': sha256(model_directory(c, merged) / 'origin.json')}
    if c.get('grpo_start') == 'sft_adapter':
        result['sft_source_sha256'] = sha256(path(c['sft_output'])/'adapter/adapter_model.safetensors')
    return result


@contextmanager
def reference_policy(model, unmerged):
    if not unmerged:
        with model.disable_adapter():
            yield
        return
    model.set_adapter('reference')
    # PEFT set_adapter can mark the active adapter trainable; keep reference frozen.
    for p in model.parameters():
        p.requires_grad_(False)
    try:
        yield
    finally:
        model.set_adapter('default')


def verify_sft_source(c, data_identity):
    if c.get('grpo_data_dir'):
        _, parent_identity = load_bundle(path(c['data_dir']))
        if data_identity['manifest'].get('parent_identity') != parent_identity:
            raise ValueError('Expanded GRPO data does not match source dataset')
        if any(data_identity['sha256'][k] != parent_identity['sha256'][k] for k in ('val', 'test')):
            raise ValueError('Expanded dataset changed held-out questions')
        data_identity = parent_identity
    directory = path(c['sft_output'])
    origin = read_json(directory/'adapter/origin.json')
    if origin.get('stage') != 'sft' or origin.get('task') != 'search' or origin.get('model_revision') != c['model_revision'] or origin.get('model_id') != c['model_id']:
        raise ValueError('Expected the original search SFT adapter')
    old = read_json(directory/'identity.json')
    if old['data'] != data_identity or old['model_origin'] != sha256(model_directory(c)/'origin.json'):
        raise ValueError('SFT source data/base differs')
    # Explicit warm start across a code change, not checkpoint resume.
    # A new GRPO budget does not change the source SFT; resume remains exact in prepare_output.
    rl_keys = ('grpo_start', 'grpo_output', 'grpo_steps', 'logprob_batch_size', 'group_size', 'rollout_batch_size',
               'grpo_groups_per_update', 'grpo_max_group_attempts', 'grpo_filter_zero_variance',
               'grpo_evidence_weight', 'grpo_protocol_weight', 'grpo_question_batch_size',
               'grpo_question_sampling', 'grpo_data_dir', 'grpo_runtime_overrides', 'grpo_compress_rollouts', 'grpo_checkpoint_every')
    current = {k:v for k,v in c.items() if k not in rl_keys}
    previous = {k:v for k,v in old['config'].items() if k not in rl_keys}
    # Explicit runtime changes for a new GRPO experiment; SFT provenance remains fixed.
    for key, change in c.get('grpo_runtime_overrides', {}).items():
        if key not in ('max_action_tokens', 'max_context', 'max_searches', 'grpo_lr', 'beta'):
            raise ValueError('Unsupported GRPO runtime override: ' + key)
        if change != {'source': old['config'].get(key), 'value': c.get(key)}:
            raise ValueError('GRPO runtime override source/value mismatch: ' + key)
        current[key] = old['config'][key]
    if current != previous or old['requirements'] != sha256('requirements-train.txt'):
        raise ValueError('SFT source config/dependencies differ')
    return directory/'adapter'


def prepare_output(out, ident, resume):
    if resume:
        if read_json(out / 'identity.json') != ident:
            raise ValueError('Config/code/data/model identity changed; cannot resume')
        if not path(resume).resolve().is_relative_to(out.resolve()):
            raise ValueError('Resume checkpoint must belong to this output directory')
    else:
        if out.exists() and any(out.iterdir()):
            raise FileExistsError('Output exists; resume or use a new output directory')
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / 'identity.json', ident)


def generate_action(model, tok, c, sample):
    import torch
    from transformers import GenerationConfig
    def generate(messages):
        prefix = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        budget = min(c['max_action_tokens'], c['max_context'] - len(prefix))
        if budget <= 0:
            return dict(text='', truncated=True, truncation_reason='context_exhausted', prefix_ids=prefix, action_ids=[])
        inputs = torch.tensor([prefix], device=model.device)
        with torch.no_grad():
            output = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                generation_config=GenerationConfig(max_new_tokens=budget, do_sample=sample,
                    temperature=1.0, top_p=1.0, top_k=0, repetition_penalty=1.0,
                    eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id, use_cache=True))
        ids = output[0, len(prefix):].tolist()
        return dict(text=tok.decode(ids, skip_special_tokens=True),
                    truncated=not ids or ids[-1] != tok.eos_token_id,
                    prefix_ids=prefix, action_ids=ids,
                    truncation_reason=None if ids and ids[-1] == tok.eos_token_id else
                        ('context_limit' if len(prefix) + len(ids) >= c['max_context'] else 'action_limit'))
    return generate


def token_logps(model, step):
    import torch
    prefix, action_ids = step['prefix_ids'], step['action_ids']
    ids = torch.tensor([prefix + action_ids], device=model.device)
    # Only materialize action-token float32 softmax, not all observation tokens.
    logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
    logits = logits[0, len(prefix)-1:-1].float()
    targets = ids[0, len(prefix):]
    return logits.log_softmax(-1).gather(-1, targets[:, None]).squeeze(-1)


def generate_action_batch(model, tok, c, sample=True, batch_size=None):
    batch_size = batch_size or c.get('rollout_batch_size', 1)
    import torch
    from transformers import GenerationConfig
    def generate(messages_batch):
        prefixes = [tok.apply_chat_template(m, tokenize=True, add_generation_prompt=True) for m in messages_batch]
        results = [None]*len(prefixes)
        # Equal remaining budgets are batched together: short contexts are never
        # truncated merely because another trajectory has a longer context.
        budgets = {}
        for i,prefix in enumerate(prefixes):
            budget = min(c['max_action_tokens'], c['max_context']-len(prefix))
            if budget <= 0:
                results[i] = dict(text='', truncated=True, truncation_reason='context_exhausted', prefix_ids=prefix, action_ids=[])
            else:
                budgets.setdefault(budget, []).append(i)
        for budget, indices in budgets.items():
            for offset in range(0,len(indices),batch_size):
                batch = indices[offset:offset+batch_size]
                width = max(len(prefixes[i]) for i in batch)
                ids = torch.tensor([[tok.pad_token_id]*(width-len(prefixes[i]))+prefixes[i] for i in batch],device=model.device)
                mask = torch.tensor([[0]*(width-len(prefixes[i]))+[1]*len(prefixes[i]) for i in batch],device=model.device)
                with torch.no_grad():
                    output = model.generate(input_ids=ids,attention_mask=mask,
                        generation_config=GenerationConfig(max_new_tokens=budget,do_sample=sample,
                            temperature=1.,top_p=1.,top_k=0,repetition_penalty=1.,
                            eos_token_id=tok.eos_token_id,pad_token_id=tok.pad_token_id,use_cache=True))
                for j,i in enumerate(batch):
                    tokens = output[j,width:].tolist()
                    ended = tok.eos_token_id in tokens
                    if ended:
                        tokens = tokens[:tokens.index(tok.eos_token_id)+1]
                    results[i] = dict(text=tok.decode(tokens,skip_special_tokens=True),truncated=not ended,
                                      prefix_ids=prefixes[i],action_ids=tokens,
                                      truncation_reason=None if ended else
                                          ('context_limit' if len(prefixes[i])+len(tokens) >= c['max_context'] else 'action_limit'))
        return results
    return generate


def batch_token_logps(model, steps, pad_id):
    """Right-pad independent causal contexts; score only their generated tokens."""
    import torch
    lengths = [len(s['prefix_ids']) + len(s['action_ids']) for s in steps]
    width = max(lengths)
    ids = torch.tensor([s['prefix_ids'] + s['action_ids'] + [pad_id]*(width-n)
                        for s,n in zip(steps,lengths)], device=model.device)
    mask = torch.tensor([[1]*n + [0]*(width-n) for n in lengths], device=model.device)
    logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
    result = []
    for i,s in enumerate(steps):
        start = len(s['prefix_ids'])
        end = lengths[i]
        scores = logits[i,start-1:end-1].float().log_softmax(-1)
        result.append(scores.gather(-1,ids[i,start:end,None]).squeeze(-1))
    return result


def policy_loss(logps, old, ref, advantage, beta, clip):
    """Per-action sum; caller normalizes by full trajectory assistant-token count."""
    import torch
    ratio = (logps - old).exp()
    task = torch.minimum(ratio * advantage, ratio.clamp(1-clip, 1+clip) * advantage)
    diff = ref - logps
    kl = diff.exp() - diff - 1
    return (-task + beta * kl).sum(), kl.detach().sum()


def adapter_origin(c, stage):
    info = {'stage': stage, 'task': 'search', 'model_id': c['model_id'], 'model_revision': c['model_revision']}
    if stage == 'grpo':
        if c.get('grpo_start') == 'sft_adapter':
            info.update(grpo_start='sft_adapter', sft_source_sha256=sha256(path(c['sft_output'])/'adapter/adapter_model.safetensors'))
        else:
            info['merged_origin_sha256'] = sha256(model_directory(c, True) / 'origin.json')
    return info


def verify_checkpoint(cp, out):
    cp = path(cp).resolve()
    if not cp.is_relative_to(out.resolve()) or '.partial' in cp.name:
        raise ValueError('Checkpoint is outside run or incomplete')
    manifest = read_json(cp / 'checkpoint-manifest.json')
    required = {'state.pt', 'adapter_model.safetensors', 'adapter_config.json', 'origin.json', 'run-identity.json'}
    if not required.issubset(manifest):
        raise ValueError('Incomplete checkpoint manifest')
    for name, digest in manifest.items():
        if Path(name).name != name or sha256(cp / name) != digest:
            raise ValueError('Corrupt search checkpoint')
    if read_json(cp / 'run-identity.json') != read_json(out / 'identity.json'):
        raise ValueError('Checkpoint belongs to a different run')


def check_lengths(c, files, tok):
    retriever = Retriever(files['corpus'], c['document_chars'])
    encoded = []
    max_action = 0
    for row in files['train']:
        for action_index, s in enumerate(sft_steps(row, retriever, c['max_searches'], c['top_k'])):
            item = encode_step(s['messages'], s['text'], tok, c['max_context'])
            action_length = sum(x != -100 for x in item['labels'])
            if action_length > c['max_action_tokens']:
                raise ValueError(f"Teacher action exceeds generation budget: {row['id']}")
            max_action = max(max_action, action_length)
            item['question_id'] = row['id']
            item['action_index'] = action_index
            encoded.append(item)
    # Initial prompts must leave at least one output token in every evaluation split.
    from search_task import initial, observation
    for split in ('val', 'test'):
        for row in files[split]:
            messages = initial(row['question'])
            messages.append({'role': 'user', 'content': observation(retriever.search(row['question'], c['top_k']))})
            if len(tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)) >= c['max_context']:
                raise ValueError(f"Initial fixed-retrieval prompt exceeds budget: {row['id']}")
    return encoded, {'sft_actions': len(encoded), 'sft_questions': len({x['question_id'] for x in encoded}),
                     'max_sft_tokens': max(len(x['input_ids']) for x in encoded),
                     'max_action_tokens': max_action, 'dynamic_rollout_lengths': 'bounded at runtime'}


def split_sft_actions(items, val_ratio, seed):
    """Hold out whole questions so teacher actions from one item never straddle splits."""
    if not 0 <= float(val_ratio) < 1:
        raise ValueError('sft_val_ratio must be in [0, 1)')
    if not items:
        raise ValueError('No SFT actions to split')
    ids = sorted({x['question_id'] for x in items})
    order = list(ids)
    random.Random(seed).shuffle(order)
    n_val = int(round(len(order) * float(val_ratio)))
    if float(val_ratio) > 0 and n_val < 1:
        n_val = 1
    if n_val >= len(order):
        raise ValueError('SFT val split would leave no training questions')
    val_ids = set(order[:n_val])
    train = [x for x in items if x['question_id'] not in val_ids]
    val = [x for x in items if x['question_id'] in val_ids]
    if not train:
        raise ValueError('Empty SFT train split')
    return train, val, {
        'val_ratio': float(val_ratio),
        'seed': seed,
        'questions_total': len(ids),
        'questions_train': len(ids) - len(val_ids),
        'questions_val': len(val_ids),
        'actions_train': len(train),
        'actions_val': len(val),
        'val_question_ids': sorted(val_ids),
    }


def sft_features(items):
    keys = ('input_ids', 'attention_mask', 'labels')
    return [{k: x[k] for k in keys} for x in items]


def sft_generation_probe(model, tok, c, files, retriever, n=16):
    """Post-SFT train-probe generation; diagnostic only, not model selection on test."""
    rows = list(files['train'])
    random.Random(int(c.get('seed', 42))).shuffle(rows)
    rows = rows[:max(0, int(n))]
    if not rows:
        return {'questions': 0}
    statuses = {}
    em = empty = unk = searches = gen_tokens = 0
    examples = []
    for row in rows:
        trace = rollout(row['question'], generate_action(model, tok, c, False), retriever,
                        c['max_searches'], c['top_k'], 'adaptive')
        status = trace.get('status', '?')
        statuses[status] = statuses.get(status, 0) + 1
        ans = (trace.get('answer') or '').strip()
        searches += int(trace.get('searches') or 0)
        gen_tokens += sum(len(s.get('action_ids') or []) for s in trace.get('steps') or [])
        correct = float(reward(trace, row['answers']))
        em += int(correct)
        if not ans:
            empty += 1
        if ans.upper() == 'UNKNOWN':
            unk += 1
        if len(examples) < 8:
            examples.append({'id': row['id'], 'status': status, 'answer': ans,
                             'gold': row['answers'], 'correct': correct,
                             'searches': trace.get('searches')})
    q = len(rows)
    return {
        'questions': q,
        'note': 'Train-probe generation diagnostics; do not treat as held-out task score.',
        'exact_match': em / q,
        'status_counts': statuses,
        'valid_finish_rate': statuses.get('answered', 0) / q,
        'invalid_action_rate': statuses.get('invalid_action', 0) / q,
        'search_limit_rate': statuses.get('search_limit', 0) / q,
        'truncated_rate': statuses.get('truncated', 0) / q,
        'empty_answer_rate': empty / q,
        'unknown_rate': unk / q,
        'avg_searches': searches / q,
        'avg_generated_tokens': gen_tokens / q,
        'examples': examples,
    }


def write_rollout(out, step, record, config):
    compressed = config.get('grpo_compress_rollouts', False)
    filename = out / (f'rollout-{step}.json' + ('.gz' if compressed else ''))
    (write_json_gzip if compressed else write_json)(filename, record)


def save_checkpoint(model, tok, optimizer, scheduler, out, step, signals, c, sampling_totals=None, sampler_state=None):
    every = c.get('grpo_checkpoint_every', 1)
    if type(every) is not int or every < 1:
        raise ValueError('grpo_checkpoint_every must be a positive integer')
    if step % every and step != c.get('grpo_steps'):
        return
    import torch
    dest = out / f'checkpoint-{step}'
    temp = out / f'checkpoint-{step}.partial-{time.time_ns()}'
    if dest.exists():
        raise FileExistsError(dest)
    temp.mkdir()
    if c.get('grpo_start') == 'sft_adapter':
        model.save_pretrained(temp, selected_adapters=['default'])
    else:
        model.save_pretrained(temp)
    tok.save_pretrained(temp)
    write_json(temp / 'run-identity.json', read_json(out / 'identity.json'))
    torch.save({'step': step, 'signals': signals, 'sampling_totals': sampling_totals, 'question_sampler': sampler_state, 'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(), 'python_rng': random.getstate(),
                'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all()}, temp / 'state.pt')
    write_json(temp / 'origin.json', adapter_origin(c, 'grpo'))
    write_json(temp / 'checkpoint-manifest.json', {p.name: sha256(p) for p in temp.iterdir() if p.is_file()})
    temp.rename(dest)


def train(c, files, data_identity, stage, resume):
    torch = require_cloud()
    from peft import get_peft_model, prepare_model_for_kbit_training
    from transformers import set_seed
    from training_visualization import Telemetry, trainer_callback
    set_seed(c['seed'])
    unmerged = stage == 'grpo' and c.get('grpo_start') == 'sft_adapter'
    merged = stage == 'grpo' and not unmerged
    source = verify_sft_source(c, data_identity) if unmerged else None
    if merged:
        base_info = read_json(model_directory(c, True) / 'origin.json')
        if base_info.get('search_data') != data_identity:
            raise ValueError('Merged model is not from this search dataset')
    out = path(c[stage + '_output'])
    ident = identity(c, data_identity, merged)
    tok = load_tokenizer(c, merged)
    retriever = Retriever(files['corpus'], c['document_chars'])
    length_files = load_bundle(path(c['data_dir']))[0] if c.get('grpo_data_dir') and stage == 'grpo' else files
    data, length_report = check_lengths(c, length_files, tok)
    if stage == 'grpo' and c.get('grpo_data_dir'):
        length_report.update(teacher_length_source=c['data_dir'], grpo_data_dir=c['grpo_data_dir'],
                             grpo_questions=len(files['train']),
                             runtime_context_policy='Each rollout stops as truncated when token budget is exhausted')
    prepare_output(out, ident, resume)
    write_json(out / 'lengths.json', length_report)
    model = prepare_model_for_kbit_training(load_model(c, merged), use_gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': True})
    if unmerged:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(source), is_trainable=True, adapter_name='default')
        model.load_adapter(str(source), adapter_name='reference', is_trainable=False)
        model.set_adapter('default')
        write_json(out / 'reference.json', {
            'policy': 'trainable copy of original SFT adapter on original NF4 base',
            'reference': 'frozen copy of original SFT adapter on the same NF4 base',
            'source_adapter_sha256': sha256(source/'adapter_model.safetensors'),
            'source_training_identity': read_json(source.parent/'identity.json'),
            'warm_start_not_checkpoint_resume': True})
    else:
        model = get_peft_model(model, lora_config(c))
    model.config.use_cache = False
    # Policy sampling and logprob scoring must agree: disable all dropout.
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    telemetry = Telemetry(out)
    try:
        if stage == 'sft':
            from datasets import Dataset
            from transformers import Trainer, TrainingArguments
            dtype = compute_dtype(c)
            micro = int(c.get('sft_micro_batch', 1))
            accum = int(c['sft_accumulation'])
            workers = int(c.get('sft_dataloader_num_workers', 2))
            val_ratio = float(c.get('sft_val_ratio', 0.1))
            gc = bool(c.get('sft_gradient_checkpointing', True))
            if micro < 1 or accum < 1 or workers < 0:
                raise ValueError('Invalid SFT batch/workers settings')
            train_items, val_items, split_report = split_sft_actions(data, val_ratio, c['seed'])
            if c.get('sft_plan'):
                import math
                plan = c['sft_plan']
                if (accum != 1 or plan['data_identity'] != data_identity or plan['split'] != split_report
                        or c['sft_steps'] != plan['epochs'] * math.ceil(len(train_items) / micro)
                        or plan['total_steps'] != c['sft_steps']):
                    raise ValueError('SFT epoch plan/data changed; regenerate the fresh SFT config')
            eval_steps = int(c.get('sft_eval_steps', max(10, c['sft_steps'] // 8)))
            if eval_steps < 1:
                raise ValueError('sft_eval_steps must be positive')
            train_ds = Dataset.from_list(sft_features(train_items))
            eval_ds = Dataset.from_list(sft_features(val_items)) if val_items else None
            load_best = bool(eval_ds is not None and c.get('sft_load_best', True))
            ta = dict(output_dir=str(out), seed=c['seed'], data_seed=c['seed'],
                bf16=dtype == torch.bfloat16, fp16=dtype == torch.float16,
                per_device_train_batch_size=micro, per_device_eval_batch_size=micro,
                gradient_accumulation_steps=accum,
                max_steps=c['sft_steps'], learning_rate=c['sft_lr'],
                warmup_ratio=float(c.get('sft_warmup_ratio', 0.0)),
                eval_strategy='steps' if eval_ds is not None else 'no',
                save_strategy='steps', save_steps=eval_steps,
                save_total_limit=int(c.get('sft_save_total_limit', 2)),
                logging_steps=1, report_to='none', remove_unused_columns=False,
                dataloader_num_workers=workers,
                gradient_checkpointing=gc,
                load_best_model_at_end=load_best)
            if eval_ds is not None:
                ta['eval_steps'] = eval_steps
            if load_best:
                ta['metric_for_best_model'] = 'eval_loss'
                ta['greater_is_better'] = False
            if gc:
                ta['gradient_checkpointing_kwargs'] = {'use_reentrant': False}
            args = TrainingArguments(**ta)
            trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds,
                processing_class=tok, data_collator=CompletionCollator(tok.pad_token_id),
                callbacks=[trainer_callback(telemetry)])
            if resume:
                for name in ('trainer_state.json', 'optimizer.pt', 'scheduler.pt', 'rng_state.pth'):
                    if not (path(resume) / name).is_file():
                        raise ValueError('Incomplete SFT checkpoint: ' + name)
            result = trainer.train(resume_from_checkpoint=str(path(resume)) if resume else None)
            final_eval = None
            if eval_ds is not None:
                final_eval = trainer.evaluate()
                telemetry.scalars({'sft_eval/'+k.removeprefix('eval_'): v for k, v in final_eval.items()
                                   if isinstance(v, (int, float))}, c['sft_steps'])
            trainer.save_model(str(out / 'adapter')); tok.save_pretrained(out / 'adapter')
            write_json(out / 'adapter/origin.json', adapter_origin(c, stage))
            probe = sft_generation_probe(model, tok, c, files, retriever,
                                         n=int(c.get('sft_probe_questions', 16)))
            gpu_peak = None
            if torch.cuda.is_available():
                gpu_peak = {
                    'max_memory_allocated_bytes': int(torch.cuda.max_memory_allocated()),
                    'max_memory_reserved_bytes': int(torch.cuda.max_memory_reserved()),
                    'device_name': torch.cuda.get_device_name(0),
                    'device_total_bytes': int(torch.cuda.get_device_properties(0).total_memory),
                }
                if gpu_peak['device_total_bytes']:
                    gpu_peak['peak_alloc_fraction'] = (
                        gpu_peak['max_memory_allocated_bytes'] / gpu_peak['device_total_bytes'])
            metrics = dict(result.metrics or {})
            if final_eval:
                metrics.update(final_eval)
            batch_info = {
                'micro_batch': micro,
                'gradient_accumulation': accum,
                'effective_batch': micro * accum,
                'dataloader_num_workers': workers,
                'gradient_checkpointing': gc,
                'warmup_ratio': float(c.get('sft_warmup_ratio', 0.0)),
                'eval_steps': eval_steps if eval_ds is not None else None,
                'load_best_model_at_end': load_best,
            }
            write_json(out / 'sft_split.json', split_report)
            write_json(out / 'result.json', {
                'status': 'finished_reload_pending',
                'metrics': metrics,
                'sft_data': split_report,
                'batch': batch_info,
                'probe': probe,
                'gpu': gpu_peak,
                'notes': {
                    'val_loss': 'Teacher-forcing loss on held-out train questions only; not HotpotQA val EM.',
                    'probe': 'Generation diagnostics on train questions; selection metric remains val adaptive EM.',
                    'gpu': 'Peak CUDA memory during SFT; TensorBoard also samples nvidia-smi utilization.',
                },
            })
            telemetry.scalars({
                'sft/effective_batch': batch_info['effective_batch'],
                'sft/actions_train': split_report['actions_train'],
                'sft/actions_val': split_report['actions_val'],
                'sft/probe_em': probe.get('exact_match') or 0,
                'sft/probe_valid_finish_rate': probe.get('valid_finish_rate') or 0,
                'sft/probe_unknown_rate': probe.get('unknown_rate') or 0,
            }, c['sft_steps'])
            return
        if compute_dtype(c) != torch.bfloat16:
            raise ValueError('Serial GRPO backend currently requires bf16; fp16 scaler is not implemented')
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=c['grpo_lr'], weight_decay=0)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        start, signals = 0, 0
        target_groups, max_attempts, filter_zero = sampling_options(c)
        evidence_weight, protocol_weight = reward_weights(c)
        sampling_totals = dict(sampled_groups=0, effective_groups=0, sampled_trajectories=0,
                               sampled_generated_tokens=0, optimizer_updates=0, skipped_rounds=0,
                               em_effective_groups=0, auxiliary_only_groups=0,
                               evidence_effective_groups=0, protocol_effective_groups=0)
        sampler = QuestionSampler(files['train'], c['seed']) if c.get('grpo_question_sampling') == 'epoch' else None
        if resume:
            cp = path(resume)
            verify_checkpoint(cp, out)
            from peft import set_peft_model_state_dict
            from peft.utils.save_and_load import load_peft_weights
            set_peft_model_state_dict(model, load_peft_weights(str(cp), device='cpu'))
            # Only our own checksummed local checkpoint; never load untrusted pickle files.
            state = torch.load(cp / 'state.pt', map_location='cpu', weights_only=False)
            optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler'])
            random.setstate(state['python_rng']); torch.set_rng_state(state['torch_rng'])
            torch.cuda.set_rng_state_all(state['cuda_rng'])
            start, signals = state['step'], state['signals']
            if sampler is not None:
                sampler.load_state_dict(state['question_sampler'])
            if state.get('sampling_totals'):
                sampling_totals.update(state['sampling_totals'])
            if cp.name != f'checkpoint-{start}' or not 0 <= start <= c['grpo_steps']:
                raise ValueError('Checkpoint step mismatch')
        for step_num in range(start, c['grpo_steps']):
            started = time.perf_counter()
            model.eval()
            def sample_group(row):
                if c.get('rollout_batch_size', 1) > 1:
                    return parallel_rollouts(row['question'], generate_action_batch(model,tok,c), retriever,
                        c['group_size'],c['max_searches'],c['top_k'])
                return [rollout(row['question'], generate_action(model, tok, c, True), retriever,
                                c['max_searches'], c['top_k']) for _ in range(c['group_size'])]
            def sample_questions(rows):
                return cross_question_rollouts(rows, generate_action_batch(model, tok, c), retriever,
                    c['group_size'], c['max_searches'], c['top_k'], c.get('rollout_batch_size', 1))
            groups, collection = collect_groups(files['train'], sample_group, target_groups,
                max_attempts, filter_zero, group_size=c['group_size'],
                evidence_weight=evidence_weight, protocol_weight=protocol_weight,
                sample_batch=sample_questions if c.get('grpo_question_batch_size', 1) > 1 else None,
                question_batch_size=c.get('grpo_question_batch_size', 1), sampler=sampler)
            if sampler is not None:
                collection.update(unique_questions_seen=sum(v > 0 for v in sampler.visits),
                                  question_coverage=sum(v > 0 for v in sampler.visits) / len(sampler.visits))
            traces = [t for g in groups for t in g['traces']]
            rewards = [r for g in groups for r in g['rewards']]
            adv = [a for g in groups for a in g['advantages']]
            # Diagnostic mean of QUESTION-LOCAL standard deviations, never pooled advantages.
            std = sum(g['reward_std'] for g in groups) / len(groups)
            collection['sampled_generated_tokens'] = sum(len(s['action_ids']) for t in traces for s in t['steps'])
            for key in ('sampled_groups', 'effective_groups', 'sampled_trajectories', 'sampled_generated_tokens',
                        'em_effective_groups', 'auxiliary_only_groups', 'evidence_effective_groups', 'protocol_effective_groups'):
                sampling_totals[key] += collection[key]
            rollout_seconds = time.perf_counter() - started
            update_started = time.perf_counter()
            batch_size = c.get('logprob_batch_size', 1)
            work, trajectory_count = training_work(groups)
            base_metrics = {**collection, 'reward': sum(rewards)/len(rewards), 'reward_std': std,
                'searches': sum(t['searches'] for t in traces)/len(traces),
                'rollout_seconds': rollout_seconds,
                'trajectories_per_second': len(traces)/max(rollout_seconds,1e-9),
                'context_truncated_trajectories': sum(any(s.get('truncation_reason') in
                    ('context_exhausted', 'context_limit') for s in t['steps']) for t in traces),
                'action_truncated_trajectories': sum(any(s.get('truncation_reason') == 'action_limit'
                    for s in t['steps']) for t in traces)}
            record = {'groups': groups, 'sampling_totals': sampling_totals}
            # Retain the legacy single-question format for old diagnostic consumers.
            if len(groups) == 1:
                record.update(sample_id=groups[0]['sample_id'], rewards=rewards, advantages=adv, traces=traces)
            if trajectory_count == 0:
                sampling_totals['skipped_rounds'] += 1
                metrics = {**base_metrics, 'optimizer_update': 0, 'signal_updates': signals,
                    'loss': 0., 'gradient_norm': 0., 'kl_per_token': 0., 'update_seconds': 0.,
                    'seconds': time.perf_counter()-started}
                telemetry.scalars({'train/'+k:v for k,v in metrics.items()}, step_num+1)
                write_rollout(out, step_num+1, {**record, 'metrics': metrics}, c)
                save_checkpoint(model, tok, optimizer, scheduler, out, step_num+1, signals, c, sampling_totals, sampler.state_dict() if sampler else None)
                print(json.dumps({'step': step_num+1, **metrics}), flush=True)
                continue
            batches = [work[i:i+batch_size] for i in range(0,len(work),batch_size)]
            # Capture behavior and frozen merged-SFT reference before updating any weights.
            with torch.no_grad():
                for batch in batches:
                    steps = [x[0] for x in batch]
                    for s,logps in zip(steps,batch_token_logps(model,steps,tok.pad_token_id)):
                        s['old'] = logps.cpu()
                    with reference_policy(model, unmerged):
                        for s,logps in zip(steps,batch_token_logps(model,steps,tok.pad_token_id)):
                            s['ref'] = logps.cpu()
            optimizer.zero_grad(set_to_none=True)
            model.train()
            kl_sum, loss_sum, valid_tokens = 0.0, 0.0, 0
            for batch in batches:
                batch_loss = None
                scores = batch_token_logps(model, [x[0] for x in batch], tok.pad_token_id)
                for (s,a,count),logps in zip(batch,scores):
                    valid_tokens += len(s['action_ids'])
                    loss, kl = policy_loss(logps, s['old'].to(logps.device), s['ref'].to(logps.device),
                                           a, c['beta'], c['clip'])
                    loss = loss / (count * trajectory_count)
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Nonfinite GRPO loss')
                    batch_loss = loss if batch_loss is None else batch_loss + loss
                    loss_sum += loss.item(); kl_sum += kl.item()
                batch_loss.backward()
                del scores, batch_loss, loss, logps
            if valid_tokens == 0:
                raise ValueError('No generated tokens; increase context budget or shorten observations')
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step(); scheduler.step()
            sampling_totals['optimizer_updates'] += 1
            signals += int(any(g['selected'] and g['has_signal'] for g in groups) and float(norm) > 0)
            metrics = {**base_metrics, 'optimizer_update': 1, 'loss': loss_sum,
                'kl_per_token': kl_sum/valid_tokens, 'gradient_norm': float(norm), 'signal_updates': signals,
                'searches': sum(t['searches'] for t in traces)/len(traces),
                'rollout_seconds': rollout_seconds, 'update_seconds': time.perf_counter()-update_started,
                'logprob_batch_size': batch_size,
                'group_size': c['group_size'], 'rollout_batch_size': c.get('rollout_batch_size',1),
                'trajectories_per_second': len(traces)/max(rollout_seconds,1e-9),
                'seconds': time.perf_counter()-started, 'peak_vram_bytes': torch.cuda.max_memory_allocated()}
            telemetry.scalars({'train/'+k:v for k,v in metrics.items()}, step_num+1)
            for t in traces:
                for s in t['steps']:
                    s.pop('old', None); s.pop('ref', None)
            write_rollout(out, step_num+1, {**record, 'metrics': metrics}, c)
            save_checkpoint(model, tok, optimizer, scheduler, out, step_num+1, signals, c, sampling_totals, sampler.state_dict() if sampler else None)
            print(json.dumps({'step': step_num+1, **metrics}), flush=True)
        model.save_pretrained(out / 'adapter', selected_adapters=['default']); tok.save_pretrained(out / 'adapter')
        write_json(out / 'adapter/origin.json', adapter_origin(c, stage))
        write_json(out / 'result.json', {'status': 'finished_reload_pending', 'signal_updates': signals,
            'sampling_totals': sampling_totals,
            'effective_group_fraction': sampling_totals['effective_groups']/max(1, sampling_totals['sampled_groups']),
            'signal_gate_passed': signals >= c['minimum_signal_updates'], 'smoke_only': c['smoke'],
            'note': 'Signal includes enabled auxiliary advantages; nonzero gradient does not prove EM improvement.'})
    except Exception as exc:
        write_json(out / 'failure.json', {'error': repr(exc), 'stage': stage})
        raise
    finally:
        telemetry.close()


def evaluate(c, files, data_identity, args):
    require_cloud()
    unmerged = args.variant == 'B3' and c.get('grpo_start') == 'sft_adapter'
    merged = args.variant in ('B1-init', 'B3') and not unmerged
    adapter = c['sft_output']+'/adapter' if args.variant == 'B1' else c['grpo_output']+'/adapter' if args.variant == 'B3' else None
    if adapter and read_json(path(adapter)/'origin.json').get('task') != 'search':
        raise ValueError('Expected search adapter')
    if args.variant == 'B1':
        verify_sft_source(c, data_identity)
    if adapter and args.variant != 'B1' and read_json(path(adapter).parent/'identity.json') != identity(c, data_identity, merged):
        raise ValueError('Adapter training identity differs from evaluation')
    if merged and read_json(model_directory(c, True)/'origin.json').get('search_data') != data_identity:
        raise ValueError('Merged search data identity mismatch')
    label = args.variant + ('-unmerged' if unmerged else '')
    if unmerged and c.get('logprob_batch_size', 1) != 1:
        label += f"-batch{c['logprob_batch_size']}"
    if c.get('rollout_batch_size',1) > 1:
        label += f"-g{c['group_size']}-r{c['rollout_batch_size']}"
    target, attempts, filtering = sampling_options(c)
    if target != 1 or attempts != 1 or filtering:
        label += f'-m{target}-a{attempts}' + ('-filtered' if filtering else '')
    evidence_weight, protocol_weight = reward_weights(c)
    if evidence_weight or protocol_weight:
        label += f'-e{evidence_weight:g}-p{protocol_weight:g}'
    output = path(args.output or f"reports/search/{label}-{args.mode}-{args.split}.json")
    if output.exists():
        raise FileExistsError('Evaluation output already exists')
    tok = load_tokenizer(c, merged)
    if unmerged:
        verify_sft_source(c, data_identity)
        info = read_json(path(adapter)/'origin.json')
        if info.get('grpo_start') != 'sft_adapter' or info.get('sft_source_sha256') != sha256(path(c['sft_output'])/'adapter/adapter_model.safetensors'):
            raise ValueError('Unmerged GRPO lineage mismatch')
        from peft import PeftModel
        model = PeftModel.from_pretrained(load_model(c), str(path(adapter)), is_trainable=False)
    else:
        model = load_model(c, merged, adapter)
    model.eval()
    retriever = Retriever(files['corpus'], c['document_chars'])
    rows = []
    eval_batch = getattr(args, 'eval_batch_size', 1)
    if eval_batch < 1:
        raise ValueError('eval_batch_size must be positive')
    started = time.perf_counter()
    traces = batched_rollouts([r['question'] for r in files[args.split]],
        generate_action_batch(model, tok, c, sample=False, batch_size=eval_batch), retriever,
        c['max_searches'], c['top_k'], args.mode, max_active=eval_batch)
    elapsed = time.perf_counter() - started
    for r, trace in zip(files[args.split], traces):
        hits = set(trace['document_ids']); support = set(r.get('support_ids', []))
        rows.append({'id': r['id'], 'correct': reward(trace, r['answers']), 'trace': trace,
                     'seconds': trace['latency_seconds'],
                     'generated_tokens': sum(len(s['action_ids']) for s in trace['steps']),
                     'support_recall': len(hits & support)/len(support) if support else None})
    print(f'Evaluated {len(rows)} questions in {elapsed:.3f}s', flush=True)
    write_json(output, {'identity': identity(c, data_identity, merged),
        'model_provenance': read_json(model_directory(c, merged)/'origin.json'), 'variant': label,
        'variant_base': args.variant,
        'mode': args.mode, 'split': args.split, 'adapter_sha256': sha256(path(adapter)/'adapter_model.safetensors') if adapter else None,
        'exact_match': sum(r['correct'] for r in rows)/len(rows), 'rows': rows,
        'execution': {'eval_batch_size': eval_batch, 'decoding': 'greedy',
                      'wall_seconds': elapsed, 'questions_per_second': len(rows)/elapsed,
                      'amortized_seconds_per_question': elapsed/len(rows),
                      'row_seconds': 'latency from queue admission; overlaps across questions'}})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['validate', 'lengths', 'download', 'sft', 'merge', 'grpo', 'eval'])
    p.add_argument('--config', default='search/config.json')
    p.add_argument('--resume')
    p.add_argument('--unmerged-sft', action='store_true', help='GRPO warm start from the original SFT adapter, with frozen SFT reference')
    p.add_argument('--logprob-batch-size', type=int, default=1, help='Concurrent action contexts in GRPO scoring/backward')
    p.add_argument('--group-size', type=int, help='Trajectories sampled for each question')
    p.add_argument('--rollout-batch-size', type=int, default=1, help='Concurrent live trajectories during generation')
    p.add_argument('--eval-batch-size', type=int, default=8, help='Concurrent greedy validation/test trajectories')
    p.add_argument('--allow-demo', action='store_true', help='Only for smoke tests; never evidence of task performance')
    p.add_argument('--variant', choices=['B0', 'B1', 'B1-init', 'B3'], default='B0')
    p.add_argument('--mode', choices=['none', 'fixed', 'adaptive'], default='adaptive')
    p.add_argument('--split', choices=['val', 'test'], default='val')
    p.add_argument('--final-test', action='store_true')
    p.add_argument('--output')
    args = p.parse_args()
    c = read_json(args.config)
    sampling_options(c)
    reward_weights(c)
    if args.eval_batch_size < 1:
        p.error('--eval-batch-size must be positive')
    if args.group_size is not None or args.rollout_batch_size != 1:
        if not args.unmerged_sft:
            p.error('Sampling overrides require --unmerged-sft')
        group = args.group_size if args.group_size is not None else c['group_size']
        if not 2 <= group <= 32 or not 1 <= args.rollout_batch_size <= group * c.get('grpo_question_batch_size', 1):
            p.error('Require 2 <= group-size <= 32 and 1 <= rollout-batch-size <= group-size * question-batch-size')
        c = {**c,'group_size':group,'rollout_batch_size':args.rollout_batch_size,
             'grpo_output':c['grpo_output']+f'-g{group}-r{args.rollout_batch_size}'}
    if not 1 <= args.logprob_batch_size <= 16:
        p.error('--logprob-batch-size must be between 1 and 16')
    if args.logprob_batch_size != 1:
        if not args.unmerged_sft:
            p.error('Batch tuning currently requires --unmerged-sft')
        c = {**c, 'logprob_batch_size': args.logprob_batch_size,
             'grpo_output': c['grpo_output']+f'-batch{args.logprob_batch_size}'}
    if args.unmerged_sft:
        if args.command != 'grpo' and not (args.command == 'eval' and args.variant == 'B3'):
            p.error('--unmerged-sft applies only to grpo or eval --variant B3')
        c = {**c, 'grpo_start': 'sft_adapter', 'grpo_output': c['grpo_output']+'-unmerged'}
    import re
    if c['model_id'] != 'Qwen/Qwen3-4B-Instruct-2507' or not re.fullmatch('[0-9a-f]{40}', c['model_revision']):
        raise ValueError('Expected pinned Qwen3-4B-Instruct-2507')
    if c['group_size'] < 2 or c['max_searches'] < 0 or c['max_context'] <= c['max_action_tokens']:
        raise ValueError('Invalid search rollout budgets')
    if min(c['top_k'], c['document_chars'], c['max_action_tokens'], c['sft_steps'], c['grpo_steps']) < 1:
        raise ValueError('Budgets and steps must be positive')
    if int(c.get('sft_micro_batch', 1)) < 1 or int(c['sft_accumulation']) < 1:
        raise ValueError('SFT batch sizes must be positive')
    if not 0 <= float(c.get('sft_val_ratio', 0.1)) < 1:
        raise ValueError('sft_val_ratio must be in [0, 1)')
    if int(c.get('sft_dataloader_num_workers', 2)) < 0:
        raise ValueError('sft_dataloader_num_workers must be >= 0')
    if c['beta'] < 0 or not 0 < c['clip'] < 1 or c['lora']['lora_dropout'] != 0:
        raise ValueError('Invalid GRPO regularization/dropout')
    if args.command == 'download':
        require_cloud()
        if (path(c['model_path'])/'origin.json').is_file():
            directory = model_directory(c)
            info = read_json(directory/'origin.json')
            if not info.get('files'):
                raise ValueError('Existing model has no recorded weight digests')
            for name, item in info['files'].items():
                if Path(name).name != name or sha256(directory/name) != item['sha256']:
                    raise ValueError('Existing model weights failed digest verification')
            print('Verified existing pinned model weights; no download needed', flush=True)
            return
        from cloud import download
        download(c); return
    use_grpo_data = bool(c.get('grpo_data_dir')) and args.command in ('grpo', 'eval', 'validate')
    if use_grpo_data and args.command == 'grpo' and c.get('grpo_start') != 'sft_adapter':
        raise ValueError('Expanded GRPO dataset requires unmerged SFT adapter')
    files, data_identity = load_bundle(path(c['grpo_data_dir'] if use_grpo_data else c['data_dir']))
    retriever = Retriever(files['corpus'], c['document_chars'])
    if not use_grpo_data:
        for row in files['train']:
            sft_steps(row, retriever, c['max_searches'], c['top_k'])
    if args.command == 'validate':
        print(json.dumps({'status': 'data_contract_passed_no_model_loaded', 'counts': {k:len(v) for k,v in files.items()},
                          'data_identity': data_identity}, ensure_ascii=False, indent=2)); return
    if data_identity['manifest'].get('demo') and not (args.allow_demo and c['smoke']):
        raise ValueError('Demo data requires --allow-demo and smoke=true; supply real data for formal runs')
    if args.command == 'lengths':
        require_cloud()
        _, report = check_lengths(c, files, load_tokenizer(c))
        print(json.dumps(report), flush=True)
        return
    if args.command in ('sft', 'grpo'):
        train(c, files, data_identity, args.command, args.resume)
    elif args.command == 'merge':
        require_cloud()
        origin = read_json(path(c['sft_output'])/'adapter/origin.json')
        if origin.get('stage') != 'sft' or origin.get('task') != 'search':
            raise ValueError('Merge requires a search SFT adapter')
        if read_json(path(c['sft_output'])/'identity.json') != identity(c, data_identity, False):
            raise ValueError('SFT config/code/data changed before merge')
        from cloud import merge
        merge(c, c['sft_output']+'/adapter')
        info = read_json(path(c['merged_model_path'])/'origin.json')
        info.update(task='search', search_data=data_identity)
        write_json(path(c['merged_model_path'])/'origin.json', info)
    else:
        if args.split == 'test' and not args.final_test:
            raise ValueError('Test requires --final-test after validation decisions are frozen')
        evaluate(c, files, data_identity, args)


if __name__ == '__main__':
    main()
