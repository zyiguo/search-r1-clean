import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from search_task import (Retriever, action, advantages, encode_step, initial, load_bundle,
                         reward, rollout, sft_steps)
from search_train import policy_loss, prepare_output, token_logps


def _torch_or_skip():
    try:
        import torch as _torch
        return _torch
    except ModuleNotFoundError:
        raise unittest.SkipTest('Optional local torch tensor checks')


class FakeTokenizer:
    eos_token_id = 9
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        tokens = []
        for m in messages:
            tokens += [2] + [ord(c)+20 for c in m['role']+':'+m['content']] + [9]
        if add_generation_prompt:
            tokens += [2] + [ord(c)+20 for c in 'assistant:']
        return tokens


class SearchTests(unittest.TestCase):
    def test_parallel_rollouts_stop_finished_and_isolate_observations(self):
        from search_task import parallel_rollouts
        retriever = Retriever([{'id':'a','contents':'alpha evidence'},{'id':'b','contents':'beta evidence'}])
        sizes = []
        def generate(batch):
            sizes.append(len(batch))
            if len(sizes)==1:
                return [{'text':'{"answer":"done"}','truncated':False},
                        {'text':'{"search":"alpha"}','truncated':False},
                        {'text':'bad','truncated':False}]
            self.assertIn('alpha evidence',batch[0][-1]['content'])
            self.assertNotIn('beta evidence',batch[0][-1]['content'])
            return [{'text':'{"answer":"alpha"}','truncated':False}]
        traces=parallel_rollouts('q',generate,retriever,3,2,1)
        self.assertEqual(sizes,[3,1])
        self.assertEqual([t['status'] for t in traces],['answered','answered','invalid_action'])
        self.assertEqual([t['searches'] for t in traces],[0,1,0])

    def test_parallel_generation_removes_eos_padding_and_preserves_prefix(self):
        torch = _torch_or_skip()
        import types
        from search_train import generate_action_batch
        class Tok:
            pad_token_id=0
            eos_token_id=9
            def apply_chat_template(self,messages,**kwargs): return messages
            def decode(self,ids,**kwargs): return str(ids)
        class Model:
            device='cpu'
            def generate(self,input_ids,attention_mask,generation_config):
                self.ids=input_ids.tolist(); self.mask=attention_mask.tolist()
                return torch.cat([input_ids,torch.tensor([[4,9,0],[5,6,7]])],dim=1)
        model=Model()
        stub=types.SimpleNamespace(GenerationConfig=lambda **kwargs:kwargs)
        with patch.dict('sys.modules',{'transformers':stub}):
            result=generate_action_batch(model,Tok(),{'max_action_tokens':3,'max_context':20,'rollout_batch_size':8})([[1],[1,2]])
        self.assertEqual(model.ids,[[0,1],[1,2]])
        self.assertEqual(model.mask,[[0,1],[1,1]])
        self.assertEqual(result[0]['prefix_ids'],[1])
        self.assertEqual(result[0]['action_ids'],[4,9])
        self.assertFalse(result[0]['truncated'])
        self.assertTrue(result[1]['truncated'])

    def test_batched_logprobs_and_gradients_match_serial_with_padding(self):
        torch = _torch_or_skip()
        from types import SimpleNamespace
        from search_train import batch_token_logps, token_logps
        class Model(torch.nn.Module):
            device = 'cpu'
            def __init__(self):
                super().__init__()
                self.emb = torch.nn.Embedding(16,8)
                self.head = torch.nn.Linear(8,16)
            def forward(self,input_ids,attention_mask,use_cache):
                # Causal dependence on all preceding unmasked tokens.
                h = (self.emb(input_ids)*attention_mask[:,:,None]).cumsum(1)
                return SimpleNamespace(logits=self.head(h))
        torch.manual_seed(4)
        m = Model()
        steps = [{'prefix_ids':[1,2], 'action_ids':[3,4]},
                 {'prefix_ids':[5,6,7], 'action_ids':[8]}]
        single = [token_logps(m,s) for s in steps]
        (-single[0].mean()/2-single[1].mean()/2).backward()
        grads = [p.grad.clone() for p in m.parameters()]
        m.zero_grad()
        batch = batch_token_logps(m,steps,0)
        for a,b in zip(single,batch): torch.testing.assert_close(a,b)
        (-batch[0].mean()/2-batch[1].mean()/2).backward()
        for p,g in zip(m.parameters(),grads): torch.testing.assert_close(p.grad,g)

    def test_unmerged_reference_is_frozen_and_policy_restored(self):
        torch = _torch_or_skip()
        from search_train import reference_policy
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.policy = torch.nn.Parameter(torch.tensor(1.))
                self.reference = torch.nn.Parameter(torch.tensor(1.), requires_grad=False)
            def set_adapter(self, name):
                self.active = name
                self.policy.requires_grad_(name == 'default')
                self.reference.requires_grad_(name == 'reference')
        m = Model(); m.set_adapter('default')
        optimizer = torch.optim.SGD([m.policy], lr=.1)
        with reference_policy(m, True):
            self.assertEqual(m.active, 'reference')
            self.assertFalse(any(p.requires_grad for p in m.parameters()))
        self.assertEqual(m.active, 'default')
        m.policy.square().backward(); optimizer.step()
        self.assertAlmostEqual(m.reference.item(), 1.)
        self.assertLess(m.policy.item(), 1.)
        with self.assertRaises(RuntimeError):
            with reference_policy(m, True): raise RuntimeError('test')
        self.assertTrue(m.policy.requires_grad)
        self.assertFalse(m.reference.requires_grad)

    def test_search_import_without_hotel_assets(self):
        import shutil
        import subprocess
        import sys
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as d:
            for name in ('common.py', 'training_common.py', 'search_task.py', 'search_train.py',
                         'cloud.py', 'hardware_probe.py', 'grpo_sampling.py', 'requirements-train.txt'):
                src = root / name
                if src.is_file():
                    shutil.copy(src, d)
            (Path(d) / 'search').mkdir(exist_ok=True)
            result = subprocess.run([sys.executable, '-c',
                'import search_train, cloud, hardware_probe; from common import path, provenance; print(path("x"))'],
                cwd=d, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def setUp(self):
        self.files, _ = load_bundle(Path(__file__).resolve().parents[1]/'search/demo')
        self.retriever = Retriever(self.files['corpus'])

    def test_strict_actions(self):
        for text in ('{"search":"a","search":"b"}', '{"answer":1}', '[]',
                     'prefix {"answer":"a"}', '{"search":"x","answer":"y"}'):
            with self.assertRaises(ValueError): action(text)

    def test_multihop_replay_and_reward(self):
        steps = sft_steps(self.files['train'][0], self.retriever, 2, 2)
        self.assertEqual(len(steps), 3)
        self.assertIn('Beacon', steps[1]['messages'][-1]['content'])
        self.assertNotIn('answers', json.dumps(steps[0]['messages']))

    def test_search_budget_and_truncation(self):
        search = lambda _: {'text':'{"search":"Atlas"}', 'truncated':False}
        t = rollout('question', search, self.retriever, 1)
        self.assertEqual(t['status'], 'search_limit')
        self.assertEqual(t['searches'], 1)
        self.assertEqual(reward(t, ['']), 0)
        t = rollout('q', lambda _:dict(text='{"answer":"12"}',truncated=True), self.retriever)
        self.assertEqual(reward(t, ['12']), 0)

    def test_fixed_and_none(self):
        def answer(messages):
            return dict(text='{"answer":"12"}',truncated=False)
        self.assertEqual(rollout('Atlas',answer,self.retriever,mode='fixed')['searches'],1)
        self.assertEqual(rollout('Atlas',answer,self.retriever,mode='none')['searches'],0)

    def test_empty_search_stable_results(self):
        self.assertEqual(self.retriever.search('zzzznonexistent'), [])
        self.assertEqual(self.retriever.search('Atlas'), self.retriever.search('Atlas'))

    def test_sft_observation_and_previous_action_mask(self):
        steps = sft_steps(self.files['train'][0], self.retriever, 2, 2)
        tok = FakeTokenizer()
        s = steps[-1]
        encoded = encode_step(s['messages'],s['text'],tok,10000)
        head = tok.apply_chat_template(s['messages'],add_generation_prompt=True)
        self.assertEqual(encoded['labels'][:len(head)], [-100]*len(head))
        self.assertEqual(encoded['labels'][len(head):],encoded['input_ids'][len(head):])
        self.assertEqual(encoded['labels'][-1], tok.eos_token_id)
        with self.assertRaises(ValueError): encode_step(s['messages'],s['text'],tok,10)

    def test_reward_group_zero_variance(self):
        a,std = advantages([0,0]); self.assertEqual(a,[0,0]); self.assertEqual(std,0)
        a,std = advantages([0,1]); self.assertLess(a[0],0); self.assertGreater(a[1],0)

    def test_no_labels_in_generation_context(self):
        seen=[]
        def gen(messages):
            seen.append(messages)
            return dict(text='{"answer":"x"}',truncated=False)
        rollout('Visible question',gen,self.retriever)
        self.assertEqual(seen[0],initial('Visible question'))

    def test_bad_teacher_rejected(self):
        row=dict(self.files['train'][0],actions=[{'answer':'wrong'}])
        with self.assertRaises(ValueError): sft_steps(row,self.retriever,2,2)

    def test_dataset_hash_failure(self):
        import shutil
        with tempfile.TemporaryDirectory() as d:
            src=Path(__file__).resolve().parents[1]/'search/demo'
            for p in src.iterdir(): shutil.copy(p,d)
            Path(d,'train.jsonl').write_text('{}\n')
            with self.assertRaisesRegex(ValueError,'digest'): load_bundle(d)

    def test_policy_gradient_direction_and_observation_exclusion(self):
        torch = _torch_or_skip()
        from types import SimpleNamespace
        class Model(torch.nn.Module):
            device='cpu'
            def __init__(self):
                super().__init__(); self.values=torch.nn.Parameter(torch.zeros(1,5,10))
            def forward(self,**kwargs): return SimpleNamespace(logits=self.values)
        model=Model()
        logps=token_logps(model,{'prefix_ids':[1,2,3],'action_ids':[4,5]})
        loss,_=policy_loss(logps,logps.detach(),logps.detach(),1.,0.,.2)
        loss.backward()
        self.assertTrue(torch.equal(model.values.grad[:,:2],torch.zeros(1,2,10)))
        self.assertLess(model.values.grad[0,2,4].item(),0)
        self.assertLess(model.values.grad[0,3,5].item(),0)

    def test_resume_requires_matching_identity(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'run'
            prepare_output(out,{'x':1},None)
            with self.assertRaises(FileExistsError): prepare_output(out,{'x':1},None)
            with self.assertRaises(ValueError): prepare_output(out,{'x':2},out/'checkpoint-1')

    def test_checkpoint_state_roundtrip(self):
        torch = _torch_or_skip()
        from search_train import save_checkpoint
        class Saver:
            def save_pretrained(self,p): Path(p,'adapter.bin').write_bytes(b'fixture')
        layer=torch.nn.Linear(1,1)
        optimizer=torch.optim.AdamW(layer.parameters(),lr=.01)
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.)
        layer(torch.ones(1,1)).sum().backward(); optimizer.step(); scheduler.step()
        with tempfile.TemporaryDirectory() as d:
            Path(d,'identity.json').write_text('{"run":1}')
            totals = {'sampled_groups': 16, 'effective_groups': 4, 'sampled_trajectories': 128,
                      'sampled_generated_tokens': 256, 'optimizer_updates': 1, 'skipped_rounds': 0}
            with patch('torch.cuda.get_rng_state_all',return_value=[]), patch('search_train.adapter_origin',return_value={'task':'search'}):
                save_checkpoint(Saver(),Saver(),optimizer,scheduler,Path(d),1,1,{},totals)
            cp=Path(d,'checkpoint-1')
            state=torch.load(cp/'state.pt',weights_only=False)
            self.assertEqual(state['step'],1)
            self.assertEqual(state['signals'],1)
            self.assertEqual(state['sampling_totals'], totals)
            self.assertTrue(state['optimizer']['state'])
            from common import sha256
            for name,h in json.loads((cp/'checkpoint-manifest.json').read_text()).items():
                self.assertEqual(sha256(cp/name),h)

    def test_incomplete_checkpoint_manifest_rejected(self):
        from search_train import verify_checkpoint
        with tempfile.TemporaryDirectory() as d:
            cp=Path(d,'checkpoint-1'); cp.mkdir()
            (cp/'checkpoint-manifest.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'Incomplete'):
                verify_checkpoint(cp,Path(d))

    def test_split_sft_actions_is_question_disjoint(self):
        from search_train import sft_features, split_sft_actions
        items = []
        for i in range(10):
            for j in range(3):
                items.append({'question_id': f'q{i}', 'action_index': j,
                              'input_ids': [i, j], 'attention_mask': [1, 1], 'labels': [-100, j]})
        train, val, report = split_sft_actions(items, 0.2, 0)
        self.assertEqual(report['questions_total'], 10)
        self.assertEqual(report['questions_val'], 2)
        self.assertEqual(report['actions_train'] + report['actions_val'], len(items))
        self.assertEqual({x['question_id'] for x in train} & {x['question_id'] for x in val}, set())
        self.assertEqual({x['question_id'] for x in val}, set(report['val_question_ids']))
        feats = sft_features(train[:1])
        self.assertEqual(set(feats[0]), {'input_ids', 'attention_mask', 'labels'})

    def test_split_sft_actions_zero_ratio_and_bounds(self):
        from search_train import split_sft_actions
        items = [{'question_id': f'q{i}', 'input_ids': [i], 'attention_mask': [1], 'labels': [-100]} for i in range(5)]
        train, val, report = split_sft_actions(items, 0.0, 1)
        self.assertEqual(val, [])
        self.assertEqual(len(train), 5)
        self.assertEqual(report['questions_val'], 0)
        with self.assertRaises(ValueError):
            split_sft_actions(items, 1.0, 0)
        with self.assertRaises(ValueError):
            split_sft_actions([], 0.1, 0)

    def test_sft_generation_probe_summarizes_statuses(self):
        from search_train import sft_generation_probe
        from search_task import Retriever
        files = {'train': [
            {'id': 'a', 'question': 'Q1', 'answers': ['yes']},
            {'id': 'b', 'question': 'Q2', 'answers': ['no']},
        ]}
        retriever = Retriever([{'id': 'd', 'contents': 'yes'}])
        traces = [
            {'steps': [{'action_ids': [1, 2]}], 'answer': 'yes', 'status': 'answered', 'searches': 1},
            {'steps': [{'action_ids': [3]}], 'answer': 'no', 'status': 'answered', 'searches': 0},
        ]
        calls = {'n': 0}
        def fake_generate(model, tok, c, sample):
            def generate(messages):
                out = traces[calls['n'] % 2]
                calls['n'] += 1
                return {'text': json.dumps({'answer': out['answer']}), 'truncated': False,
                        'prefix_ids': [0], 'action_ids': out['steps'][0]['action_ids']}
            return generate
        c = {'max_searches': 2, 'top_k': 1, 'seed': 0, 'max_action_tokens': 8, 'max_context': 32}
        with patch('search_train.generate_action', side_effect=fake_generate):
            report = sft_generation_probe(None, None, c, files, retriever, n=2)
        self.assertEqual(report['questions'], 2)
        self.assertEqual(report['exact_match'], 1.0)
        self.assertEqual(report['status_counts'], {'answered': 2})
        self.assertEqual(report['avg_searches'], 0.0)
        self.assertEqual(report['avg_generated_tokens'], 1.5)
        self.assertIn('do not treat as held-out', report['note'])

    def test_teacher_action_budget_rejected(self):
        from search_train import check_lengths
        c={'document_chars':800,'max_searches':2,'top_k':2,'max_context':10000,'max_action_tokens':1}
        with self.assertRaisesRegex(ValueError,'generation budget'):
            check_lengths(c,self.files,FakeTokenizer())

    def test_comparison_rejects_mixed_model_origin(self):
        from search_compare import compare
        ident={'config':{},'data':{},'code':{},'requirements':'x','model_origin':'a'}
        row={'id':'x','correct':1,'trace':{'searches':1},'generated_tokens':1,'seconds':1}
        a={'identity':ident,'variant':'B0','mode':'adaptive','split':'val','exact_match':1,'rows':[row]}
        b={**a,'variant':'B1','identity':{**ident,'model_origin':'b'},'adapter_sha256':'sft'}
        with patch('search_compare.read_json',side_effect=[a,b]):
            with self.assertRaisesRegex(ValueError,'different origins'): compare(['a','b'])

    def test_hotpot_converter_keeps_gold_out_of_eval_context(self):
        from prepare_search_data import convert
        def row(n):
            return {'_id':str(n),'question':f'Which number is linked to Code{n}?','answer':str(n),
                    'context':[[f'Code{n}',[f'Code{n} uses Link{n}.']],
                               [f'Link{n}',[f'Link{n} has number {n}.']]],
                    'supporting_facts':[[f'Code{n}',0],[f'Link{n}',0]]}
        with tempfile.TemporaryDirectory() as d:
            a,b=Path(d,'train.json'),Path(d,'dev.json')
            a.write_text(json.dumps([row(1),row(2)])); b.write_text(json.dumps([row(3)]))
            convert(a,b,Path(d,'result'),1,1,1)
            data,_=load_bundle(Path(d,'result'))
            self.assertNotIn('actions',data['test'][0])
            self.assertNotIn('answers',json.dumps(data['corpus']))


if __name__ == '__main__': unittest.main()
