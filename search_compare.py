"""Compare completed search evaluations under matched config/data/code and split."""
import argparse
from common import read_json


def compare(files):
    records=[read_json(f) for f in files]
    if not records:
        raise ValueError('No reports supplied')
    first=records[0]
    origins = {}
    for r in records:
        if not r['rows'] or len({x['id'] for x in r['rows']}) != len(r['rows']):
            raise ValueError('Empty or duplicate evaluation rows')
        calculated = sum(x['correct'] for x in r['rows']) / len(r['rows'])
        if abs(calculated - r['exact_match']) > 1e-12:
            raise ValueError('Reported EM differs from rows')
        family = 'merged' if r.get('variant_base', r['variant']) in ('B1-init', 'B3') and r['identity']['config'].get('grpo_start') != 'sft_adapter' else 'base'
        origin = r['identity']['model_origin']
        if family in origins and origins[family] != origin:
            raise ValueError('Models in the same family use different origins')
        origins[family] = origin
        if r['split'] != first['split']:
            raise ValueError('Split mismatch')
        for key in ('config','data','code','requirements'):
            if r['identity'][key] != first['identity'][key]:
                raise ValueError('Comparison identity mismatch: '+key)
        if [x['id'] for x in r['rows']] != [x['id'] for x in first['rows']]:
            raise ValueError('Evaluation question IDs mismatch')
    sft_hashes = {r['adapter_sha256'] for r in records if r.get('variant_base', r['variant']) == 'B1'}
    if len(sft_hashes) > 1:
        raise ValueError('Multiple SFT adapters in comparison')
    for r in records:
        if r.get('variant_base', r['variant']) in ('B1-init', 'B3') and r['identity']['config'].get('grpo_start') != 'sft_adapter' and sft_hashes:
            if r.get('model_provenance', {}).get('adapter_sha256') not in sft_hashes:
                raise ValueError('Merged model was not built from the evaluated SFT adapter')
    lines=['| Variant | Mode | EM | Search calls | Generated tokens | Seconds/question |',
           '|---|---|---:|---:|---:|---:|']
    for r in records:
        n=len(r['rows'])
        lines.append(f"| {r['variant']} | {r['mode']} | {r['exact_match']:.3f} | "
                     f"{sum(x['trace']['searches'] for x in r['rows'])/n:.2f} | "
                     f"{sum(x['generated_tokens'] for x in r['rows'])/n:.1f} | "
                     f"{r.get('execution', {}).get('amortized_seconds_per_question', sum(x['seconds'] for x in r['rows'])/n):.2f} |")
    return '\n'.join(lines)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('reports',nargs='+')
    print(compare(p.parse_args().reports))
