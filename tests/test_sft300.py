import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from common import ROOT, read_json, write_json, sha256
from search_task import load_bundle
from scripts.prepare_sft300 import prepare


class SFTTrajectoryTests(unittest.TestCase):
    def test_adds_three_search_rule_trajectory_from_frozen_candidate_pool(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source, candidate, evaluation = root/'source', root/'candidate', root/'eval'
            for folder in (source,candidate):
                shutil.copytree(ROOT/'search/demo',folder)
            original = load_bundle(candidate)[0]['train'][0]
            extra = copy.deepcopy(original)
            extra.update(id='new-three', question='New three document question',support_ids=['d1','d2','d3'])
            extra.pop('actions')
            (candidate/'train.jsonl').write_text(json.dumps(original)+'\n'+json.dumps(extra)+'\n',encoding='utf-8')
            manifest = read_json(candidate/'manifest.json')
            manifest['sha256']['train'] = sha256(candidate/'train.jsonl')
            write_json(candidate/'manifest.json',manifest)
            _, candidate_identity = load_bundle(candidate)
            shutil.copytree(candidate,evaluation)
            manifest.update(evaluation_only=True,source_training_identity=candidate_identity)
            write_json(evaluation/'manifest.json',manifest)
            result = prepare(ROOT/'search/pilot-sft-metrics.json',source,evaluation,root/'out',root/'config.json',2,candidate)
            self.assertEqual(result['retained_original'],1)
            self.assertEqual(result['added_rule_trajectories'],1)
            files, _ = load_bundle(root/'out')
            added = next(r for r in files['train'] if r['id']=='new-three')
            self.assertEqual(len(added['actions']),4)
            self.assertEqual(sum('search' in a for a in added['actions']),3)
            self.assertEqual(added['actions'][-1],{'answer':'12'})
            self.assertEqual(files['val'],load_bundle(evaluation)[0]['val'])

    def test_complete_selection_split_and_frozen_evaluation(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source, evaluation = root/'source', root/'eval'
            for folder in (source,evaluation):
                shutil.copytree(ROOT/'search/demo',folder)
            row = load_bundle(source)[0]['train'][0]
            rows = []
            for i in range(12):
                item = copy.deepcopy(row)
                item.update(id=f'train-{i}', question=f'Question {i}')
                rows.append(item)
            rows[0]['actions'] = [{'answer':'wrong'}]
            # Valid in source, but excluded by the independent evaluation questions.
            rows[1]['question'] = 'External heldout question'
            (source/'train.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
            manifest = read_json(source/'manifest.json')
            manifest['sha256']['train'] = sha256(source/'train.jsonl')
            write_json(source/'manifest.json',manifest)
            vals = load_bundle(evaluation)[0]['val']
            vals[0]['question'] = 'External heldout question'
            (evaluation/'val.jsonl').write_text(json.dumps(vals[0])+'\n',encoding='utf-8')
            manifest = read_json(evaluation/'manifest.json')
            manifest.update(evaluation_only=True)
            manifest['sha256']['val'] = sha256(evaluation/'val.jsonl')
            write_json(evaluation/'manifest.json',manifest)
            config = ROOT/'search/pilot-sft-metrics.json'
            result = prepare(config,source,evaluation,root/'out',root/'config.json',10)
            c = read_json(root/'config.json')
            self.assertEqual(result['split']['questions_train'],9)
            self.assertEqual(result['split']['questions_val'],1)
            self.assertEqual(c['sft_steps'],6)  # 27 training actions / micro16 * 3 epochs
            self.assertEqual(c['lora']['r'],32)
            self.assertEqual(c['lora']['lora_alpha'],64)
            files, identity = load_bundle(root/'out')
            self.assertEqual(len(files['train']),10)
            self.assertNotIn('train-0',{r['id'] for r in files['train']})
            self.assertNotIn('train-1',{r['id'] for r in files['train']})
            self.assertEqual(c['sft_plan']['data_identity'],identity)
            for name in ('corpus','val','test'):
                self.assertEqual((evaluation/(name+'.jsonl')).read_bytes(),(root/'out'/(name+'.jsonl')).read_bytes())
            with self.assertRaises(FileExistsError):
                prepare(config,source,evaluation,root/'out',root/'config.json',10)
            with self.assertRaisesRegex(ValueError,'eligible complete trajectories'):
                prepare(config,source,evaluation,root/'too-many',root/'bad.json',11)
            self.assertFalse((root/'too-many').exists())
