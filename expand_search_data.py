"""Extend RL questions from official HotpotQA train; retain held-out files byte-for-byte."""
import argparse
import hashlib
import json
import random
from pathlib import Path
from common import sha256, write_json
from search_task import load_bundle, normalize


def official_row(row):
    """Normalize Hugging Face HotpotQA structs without changing their text."""
    if '_id' in row:
        return row
    context, facts = row['context'], row['supporting_facts']
    return {'_id': row['id'], 'question': row['question'], 'answer': row['answer'],
            'context': list(zip(context['title'], context['sentences'])),
            'supporting_facts': list(zip(facts['title'], facts['sent_id']))}


def read_source(source):
    if source.is_file():
        rows = json.loads(source.read_text(encoding='utf-8'))
        if not isinstance(rows, list):
            raise ValueError('Expected official HotpotQA JSON array')
        return rows, {source.name: sha256(source)}
    # The user's already downloaded distractor train shards; do not include validation.
    shards = sorted(source.glob('train-*.parquet'))
    expected = {'train-00000-of-00002.parquet', 'train-00001-of-00002.parquet'}
    if {p.name for p in shards} != expected:
        raise ValueError('Supply official train JSON or directory containing both HF distractor train shards')
    import pyarrow.parquet as pq
    rows = []
    for shard in shards:
        for batch in pq.ParquetFile(shard).iter_batches(batch_size=1024):
            rows.extend(official_row(r) for r in batch.to_pylist())
    return rows, {p.name: sha256(p) for p in shards}


def expand(source, raw_train, output, train_size=2000, seed=42):
    source, raw_train, output = map(Path, (source, raw_train, output))
    files, parent = load_bundle(source)
    if output.exists():
        raise FileExistsError('Choose a new dataset output directory')
    if train_size < len(files['train']):
        raise ValueError('Target size cannot shrink source training set')
    raw, source_hashes = read_source(raw_train)
    random.Random(seed).shuffle(raw)
    seen_ids = {r['id'] for k in ('train', 'val', 'test') for r in files[k]}
    seen_q = {normalize(r['question']) for k in ('train', 'val', 'test') for r in files[k]}
    corpus = {d['id']: d for d in files['corpus']}
    rows = list(files['train'])
    rejected = 0
    for r in raw:
        if len(rows) >= train_size:
            break
        if r['_id'] in seen_ids or normalize(r['question']) in seen_q:
            continue
        titles = list(dict.fromkeys(t for t, _ in r['supporting_facts']))
        docs, title_ids = {}, {}
        for title, sentences in r['context']:
            content = title + '\n' + ''.join(sentences)
            doc_id = hashlib.sha256(content.encode('utf-8')).hexdigest()
            docs[doc_id] = {'id': doc_id, 'contents': content}
            title_ids[title] = doc_id
        if not titles or any(t not in title_ids for t in titles) or not str(r.get('answer', '')).strip():
            rejected += 1
            continue
        corpus.update(docs)
        rows.append({'id': r['_id'], 'question': r['question'], 'answers': [r['answer']],
                     'support_ids': [title_ids[t] for t in titles]})
        seen_ids.add(r['_id']); seen_q.add(normalize(r['question']))
    if len(rows) != train_size:
        raise ValueError(f'Only {len(rows)} distinct answerable questions available')
    output.mkdir(parents=True)
    for split in ('val', 'test'):
        (output / (split + '.jsonl')).write_bytes((source / (split + '.jsonl')).read_bytes())
    for name, values in [('train', rows), ('corpus', sorted(corpus.values(), key=lambda d: d['id']))]:
        (output / (name + '.jsonl')).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in values), encoding='utf-8', newline='\n')
    manifest = {**parent['manifest'], 'parent_identity': parent, 'seed': seed,
                'sha256': {k: sha256(output / (k + '.jsonl')) for k in ('train', 'val', 'test', 'corpus')},
                'extension_source': {'files_sha256': source_hashes},
                'teacher': 'Original SFT actions retained; added RL questions have no teacher actions.',
                'extension_policy': 'Official train only; ID and normalized-question dedup; supporting documents present. Corpus extended; reevaluate SFT baseline.',
                'extension_rejected': rejected}
    write_json(output / 'manifest.json', manifest)
    load_bundle(output)
    return {'train_questions': len(rows), 'corpus_documents': len(corpus), 'output': str(output), 'rejected': rejected}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', default='search/hotpot-pilot')
    p.add_argument('--train', required=True, help='Official train JSON or HF distractor directory with both train parquet shards')
    p.add_argument('--output', default='search/hotpot-grpo-2000')
    p.add_argument('--train-size', type=int, default=2000)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    print(json.dumps(expand(a.source, a.train, a.output, a.train_size, a.seed)))


if __name__ == '__main__':
    main()
