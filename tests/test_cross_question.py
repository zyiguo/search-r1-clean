import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from search_task import Retriever, rollout, batched_rollouts, cross_question_rollouts, load_bundle
from grpo_sampling import QuestionSampler, collect_groups
from expand_search_data import expand


def generated(text):
    return dict(text=text, truncated=False, prefix_ids=[1], action_ids=[2])


class CrossQuestionTests(unittest.TestCase):
    def test_queue_refill_isolation_and_order(self):
        batches = []
        def generate(messages):
            batches.append([m[1]['content'] for m in messages])
            result = []
            for m in messages:
                q = m[1]['content']
                if q == 'slow' and len(m) == 2:
                    result.append(generated('{"search":"slow"}'))
                else:
                    if q != 'slow':
                        self.assertEqual(len(m), 2)
                    result.append(generated(json.dumps({'answer': q})))
            return result
        rows = batched_rollouts(['slow', 'fast', 'next'], generate,
                                Retriever([{'id':'s', 'contents':'slow evidence'}]), max_active=2)
        self.assertEqual(batches, [['slow', 'fast'], ['slow', 'next']])
        self.assertEqual([r['answer'] for r in rows], ['slow', 'fast', 'next'])
        self.assertEqual([r['document_ids'] for r in rows], [['s'], [], []])

    def test_serial_and_batch_modes_agree(self):
        retriever = Retriever([{'id':'a', 'contents':'question evidence'}])
        def generate(m):
            return generated('{"search":"question"}' if len(m) == 2 else '{"answer":"yes"}')
        for mode in ('none', 'fixed', 'adaptive'):
            expected = rollout('question', generate, retriever, 2, 1, mode)
            actual = batched_rollouts(['question'] * 3, lambda ms: [generate(m) for m in ms],
                                      retriever, 2, 1, mode, 2)
            for row in actual:
                row.pop('latency_seconds')
                self.assertEqual(row, expected)

    def test_cross_question_replica_grouping(self):
        batches = []
        def generate(ms):
            batches.append([m[1]['content'] for m in ms])
            return [generated(json.dumps({'answer': m[1]['content']})) for m in ms]
        rows = [{'question':'a'}, {'question':'b'}]
        groups = cross_question_rollouts(rows, generate, Retriever([]), 3, max_active=2)
        self.assertEqual(batches, [['a', 'b']] * 3)
        self.assertEqual([[t['answer'] for t in g] for g in groups], [['a'] * 3, ['b'] * 3])

    def test_batched_collection_accounts_surplus_and_local_advantages(self):
        rows = [dict(id=str(i), answers=['yes']) for i in range(4)]
        calls = []
        def sample(batch):
            calls.append(len(batch))
            return [[dict(status='answered', answer=a, document_ids=[], steps=[])
                     for a in ('yes', 'no')] for _ in batch]
        groups, metrics = collect_groups(rows, None, 1, 4, True,
                                          sample_batch=sample, question_batch_size=4)
        self.assertEqual(calls, [4])
        self.assertEqual(metrics['sampled_trajectories'], 8)
        self.assertEqual(metrics['selected_trajectories'], 2)
        self.assertEqual(metrics['surplus_groups'], 3)
        for g in groups:
            self.assertAlmostEqual(sum(g['advantages']), 0)

    def test_epoch_coverage_and_resume(self):
        rows = [dict(id=str(i)) for i in range(7)]
        sampler = QuestionSampler(rows, 42)
        first = [sampler.draw()['id'] for _ in range(7)]
        self.assertEqual(len(set(first)), 7)
        state = copy.deepcopy(sampler.state_dict())
        restored = QuestionSampler(rows, 99)
        restored.load_state_dict(state)
        self.assertEqual([sampler.draw()['id'] for _ in range(20)],
                         [restored.draw()['id'] for _ in range(20)])

    def test_epoch_boundary_does_not_drop_deferred_questions(self):
        rows = [dict(id=str(i)) for i in range(7)]
        sampler = QuestionSampler(rows, 42)
        for _ in range(10):
            seen = set()
            for _ in range(5):
                row = sampler.draw(seen)
                self.assertNotIn(row['id'], seen)
                seen.add(row['id'])
        self.assertLessEqual(max(sampler.visits) - min(sampler.visits), 1)

    def test_expansion_preserves_holdouts_and_filters_missing_evidence(self):
        original, parent = load_bundle('search/demo')
        heldout = original['val'][0]
        candidates = [dict(_id='new', question='A new question?', answer='yes',
                           context=[['Title', ['The evidence.']]], supporting_facts=[['Title', 0]]),
                      dict(_id='bad', question='Missing evidence?', answer='yes',
                           context=[], supporting_facts=[['Missing', 0]]),
                      dict(_id=heldout['id'], question=heldout['question'], answer='yes',
                           context=[], supporting_facts=[])]
        with tempfile.TemporaryDirectory() as d:
            raw, dest = Path(d)/'train.json', Path(d)/'expanded'
            raw.write_text(json.dumps(candidates))
            expand('search/demo', raw, dest, len(original['train']) + 1)
            data, ident = load_bundle(dest)
            self.assertEqual(ident['manifest']['parent_identity'], parent)
            for split in ('val', 'test'):
                self.assertEqual((dest/(split+'.jsonl')).read_bytes(),
                                 Path('search/demo', split+'.jsonl').read_bytes())
            self.assertEqual(data['train'][-1]['id'], 'new')
            self.assertNotIn('actions', data['train'][-1])
            self.assertTrue(set(data['train'][-1]['support_ids']) <= {x['id'] for x in data['corpus']})
            with self.assertRaises(FileExistsError):
                expand('search/demo', raw, dest, len(original['train']) + 1)

    def test_evaluation_uses_greedy_batches_and_wall_time(self):
        from types import SimpleNamespace
        from unittest.mock import MagicMock
        import search_train
        from common import read_json
        config = read_json('search/pilot-sft-metrics.json')
        files = {'corpus': [], 'val': [dict(id=str(i), question=str(i), answers=[str(i)]) for i in range(3)]}
        calls = []
        def factory(model, tok, c, sample=True, batch_size=None):
            self.assertFalse(sample)
            self.assertEqual(batch_size, 2)
            def generate(ms):
                calls.append(len(ms))
                return [generated(json.dumps({'answer': m[1]['content']})) for m in ms]
            return generate
        with tempfile.TemporaryDirectory() as d:
            output = Path(d)/'eval.json'
            args = SimpleNamespace(variant='B0', mode='adaptive', split='val', output=str(output), eval_batch_size=2)
            with patch.multiple(search_train, require_cloud=lambda: None,
                                load_model=lambda *a: MagicMock(), load_tokenizer=lambda *a: None,
                                model_directory=lambda *a: Path(d), read_json=lambda *a: {},
                                identity=lambda *a: {}, generate_action_batch=factory):
                search_train.evaluate(config, files, {}, args)
            report = read_json(output)
            self.assertEqual(calls, [2, 1])
            self.assertEqual(report['exact_match'], 1.)
            self.assertEqual(report['execution']['decoding'], 'greedy')
            self.assertAlmostEqual(report['execution']['amortized_seconds_per_question'] * 3,
                                   report['execution']['wall_seconds'])

    def test_hf_struct_conversion_retains_sentence_text(self):
        from expand_search_data import official_row
        row = dict(id='x', question='Q', answer='A',
                   context={'title': ['T'], 'sentences': [['first ', 'second']]},
                   supporting_facts={'title': ['T'], 'sent_id': [1]})
        converted = official_row(row)
        self.assertEqual(converted['_id'], 'x')
        self.assertEqual(converted['context'], [('T', ['first ', 'second'])])
        self.assertEqual(converted['supporting_facts'], [('T', 1)])
