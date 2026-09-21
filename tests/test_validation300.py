import copy
import json
from pathlib import Path
import tempfile
import unittest

from common import ROOT, sha256
from search_task import load_bundle
from scripts.prepare_validation300 import prepare
from scripts.eval_validation300 import verify_dataset


class ValidationExtensionTests(unittest.TestCase):
    def test_extension_preserves_original_and_excludes_duplicates(self):
        source = ROOT / 'search/demo'
        original, identity = load_bundle(source)
        def row(i, question=None):
            return {'_id': i, 'question': question or 'Question '+i,
                    'answer': 'answer', 'supporting_facts': [['Title '+i, 0]],
                    'context': [['Title '+i, ['answer']]]}
        raw = [row('new1'), row('new2'), row(original['train'][0]['id']),
               row('duplicate-question', original['train'][0]['question']),
               row('bad')]
        raw[-1]['context'] = []
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            inp = root/'raw.json'
            inp.write_text(json.dumps(raw), encoding='utf-8')
            before = {p.name: sha256(p) for p in source.iterdir() if p.is_file()}
            prepare(source, inp, root/'out', size=3)
            files, expanded = load_bundle(root/'out')
            self.assertEqual({r['id'] for r in files['val'][1:]}, {'new1','new2'})
            self.assertTrue(all('actions' not in r for r in files['val']))
            verify_dataset(original, identity, files, expanded)
            self.assertEqual(before, {p.name: sha256(p) for p in source.iterdir() if p.is_file()})
            prepare(source, inp, root/'out2', size=3)
            self.assertEqual(expanded, load_bundle(root/'out2')[1])
            changed = copy.deepcopy(expanded)
            changed['manifest']['source_training_identity'] = {}
            with self.assertRaises(ValueError):
                verify_dataset(original, identity, files, changed)
            changed_files = copy.deepcopy(files)
            changed_files['val'][0]['question'] = 'changed'
            with self.assertRaises(ValueError):
                verify_dataset(original, identity, changed_files, expanded)
            changed_files = copy.deepcopy(files)
            changed_files['corpus'] = []
            with self.assertRaises(ValueError):
                verify_dataset(original, identity, changed_files, expanded)
            with self.assertRaises(FileExistsError):
                prepare(source, inp, root/'out', size=3)
            with self.assertRaises(ValueError):
                prepare(source, inp, root/'insufficient', size=4)
            self.assertFalse((root/'insufficient').exists())


if __name__ == '__main__':
    unittest.main()
