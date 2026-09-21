"""Create a separate evaluation-only bundle, preserving the original validation subset."""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import path, sha256, write_json
from expand_search_data import read_source
from search_task import load_bundle, normalize


def prepare(source, raw_train, output, size=300, seed=20260921):
    source, raw_train, output = map(path, (source, raw_train, output))
    files, identity = load_bundle(source)
    if output.exists():
        raise FileExistsError('Use a new validation dataset directory')
    if type(size) is not int or size <= len(files['val']):
        raise ValueError('Validation target must exceed original validation size')
    raw, hashes = read_source(raw_train)
    random.Random(seed).shuffle(raw)
    seen_ids = {r['id'] for k in ('train', 'val', 'test') for r in files[k]}
    seen_questions = {normalize(r['question']) for k in ('train', 'val', 'test') for r in files[k]}
    val = list(files['val'])
    corpus = {r['id']: r for r in files['corpus']}
    rejected = 0
    for r in raw:
        if len(val) == size:
            break
        question = normalize(r['question'])
        if r['_id'] in seen_ids or question in seen_questions:
            continue
        titles = list(dict.fromkeys(t for t, _ in r['supporting_facts']))
        docs, title_ids = {}, {}
        for title, sentences in r['context']:
            text = title + '\n' + ''.join(sentences)
            did = hashlib.sha256(text.encode('utf-8')).hexdigest()
            docs[did] = {'id': did, 'contents': text}
            title_ids[title] = did
        if not titles or any(t not in title_ids for t in titles) or not r.get('answer', '').strip():
            rejected += 1
            continue
        val.append({'id': r['_id'], 'question': r['question'], 'answers': [r['answer']],
                    'support_ids': [title_ids[t] for t in titles]})
        corpus.update(docs)
        seen_ids.add(r['_id']); seen_questions.add(question)
    if len(val) != size:
        raise ValueError(f'Insufficient unseen validation questions: {len(val)}/{size}')
    output.mkdir(parents=True)
    for split in ('train', 'test'):
        (output/(split+'.jsonl')).write_bytes((source/(split+'.jsonl')).read_bytes())
    for name, rows in [('val', val), ('corpus', sorted(corpus.values(), key=lambda d: d['id']))]:
        with (output/(name+'.jsonl')).open('w', encoding='utf-8', newline='\n') as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False)+'\n')
    manifest = dict(source=identity['manifest']['source'], license=identity['manifest']['license'],
                    demo=identity['manifest'].get('demo', False), evaluation_only=True,
                    source_training_identity=identity, seed=seed, source_files=hashes,
                    original_val_ids=[r['id'] for r in files['val']],
                    validation_size=size, rejected_missing_evidence=rejected,
                    protocol='Original validation retained, unseen official-train questions added; shared corpus extended; not official leaderboard.',
                    sha256={k: sha256(output/(k+'.jsonl')) for k in ('train','val','test','corpus')})
    write_json(output/'manifest.json', manifest)
    load_bundle(output)
    return dict(directory=str(output), total=len(val), retained=len(files['val']),
                added=len(val)-len(files['val']), corpus_documents=len(corpus))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', default='search/hotpot-grpo-2000')
    p.add_argument('--train', default='search/raw-recovery/hotpotqa-hf/distractor')
    p.add_argument('--output', default='search/eval-val300')
    p.add_argument('--size', type=int, default=300)
    p.add_argument('--seed', type=int, default=20260921)
    a = p.parse_args()
    print(json.dumps(prepare(a.source, a.train, a.output, a.size, a.seed), ensure_ascii=False))


if __name__ == '__main__':
    main()
