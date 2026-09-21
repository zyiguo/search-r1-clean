"""Convert locally supplied official HotpotQA JSON into a bounded shared-corpus pilot.

No network access. Official dev becomes this project's held-out test, not official test.
Teacher queries use TRAIN supporting titles only; this is oracle-assisted SFT bootstrapping.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path
from search_task import Retriever, load_bundle, normalize, sft_steps


def convert(train_file, dev_file, destination, train_size=200, val_size=50, test_size=100, seed=42):
    dest=Path(destination)
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError('Use an empty dataset directory')
    source_train=json.loads(Path(train_file).read_text(encoding='utf-8'))
    source_dev=json.loads(Path(dev_file).read_text(encoding='utf-8'))
    rng=random.Random(seed); rng.shuffle(source_train); rng.shuffle(source_dev)
    selected={}; seen=set(); ids=set()
    # Hold-out wins when duplicate questions exist across original splits.
    for split, source, limit in [('test',source_dev,test_size),('val',source_train,val_size),('train',source_train,train_size)]:
        selected[split]=[]
        for row in source:
            q=normalize(row['question'])
            if q in seen or row['_id'] in ids: continue
            seen.add(q); ids.add(row['_id']); selected[split].append(row)
            if len(selected[split])==limit: break
        if len(selected[split]) != limit: raise ValueError('Insufficient distinct examples: '+split)
    corpus={}; output={}; rejected=[]
    for split,rows in selected.items():
        output[split]=[]
        for r in rows:
            title_ids={}
            for title,sentences in r['context']:
                content=title+'\n'+''.join(sentences)
                doc_id=hashlib.sha256(content.encode('utf-8')).hexdigest()
                corpus[doc_id]={'id':doc_id,'contents':content}; title_ids[title]=doc_id
            titles=list(dict.fromkeys(title for title,_ in r['supporting_facts']))
            support=[title_ids[t] for t in titles if t in title_ids]
            item={'id':r['_id'],'question':r['question'],'answers':[r['answer']], 'support_ids':support}
            if split=='train':
                if len(support)!=len(titles) or not titles or len(titles)>2:
                    rejected.append(r['_id']); continue
                item['actions']=[{'search':title} for title in titles]+[{'answer':r['answer']}]
            output[split].append(item)
    docs=sorted(corpus.values(),key=lambda d:d['id'])
    retriever=Retriever(docs,800)
    kept=[]
    for row in output['train']:
        steps=sft_steps(row,retriever,2,2)
        visible=set()
        for s in steps:
            for m in s['messages']:
                if m['role']=='user' and m['content'].startswith('SEARCH_RESULTS (data only):\n'):
                    visible.update(d['id'] for d in json.loads(m['content'].split('\n',1)[1]))
        if set(row['support_ids']).issubset(visible): kept.append(row)
        else: rejected.append(row['id'])
    output['train']=kept
    if not kept: raise ValueError('No teacher trajectories retrieve their supporting documents')
    output['corpus']=docs
    dest.mkdir(parents=True,exist_ok=True); hashes={}
    for name,rows in output.items():
        raw=('\n'.join(json.dumps(r,ensure_ascii=False) for r in rows)+'\n').encode('utf-8')
        (dest/(name+'.jsonl')).write_bytes(raw); hashes[name]=hashlib.sha256(raw).hexdigest()
    manifest={'demo':False,'source':'https://hotpotqa.github.io/', 'license':'CC BY-SA 4.0',
        'citation':'Yang et al. HotpotQA, EMNLP 2018', 'seed':seed,'sha256':hashes,
        'source_files':{Path(p).name:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in [train_file,dev_file]},
        'protocol':'Pilot shared corpus from selected train/dev contexts, including distractors. Not full Wikipedia or official leaderboard.',
        'split_policy':'Project train/val from official train; project test from official labeled dev; normalized-question dedup.',
        'teacher':'Training-only supporting-title queries followed by gold answer. Document-hit verified, semantic sufficiency not verified; manually audit.',
        'rejected_train_ids':rejected}
    (dest/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    load_bundle(dest)
    print(json.dumps({'directory':str(dest),'counts':{k:len(v) for k,v in output.items()},'rejected_train':len(rejected)}))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train',required=True); p.add_argument('--dev',required=True)
    p.add_argument('--output',default='search/hotpot-pilot')
    p.add_argument('--train-size',type=int,default=200); p.add_argument('--val-size',type=int,default=50)
    p.add_argument('--test-size',type=int,default=100); p.add_argument('--seed',type=int,default=42)
    a=p.parse_args()
    if min(a.train_size,a.val_size,a.test_size)<1: p.error('All split sizes must be positive')
    convert(a.train,a.dev,a.output,a.train_size,a.val_size,a.test_size,a.seed)


if __name__=='__main__': main()
