"""Arithmetic planning estimates, not GPU measurements. No model or dataset loading."""
import argparse
import json
from common import path, read_json, write_json

GIB = 1024 ** 3
OFFICIAL = {'hidden_size': 2560, 'intermediate_size': 9728, 'num_hidden_layers': 36,
            'num_attention_heads': 32, 'num_key_value_heads': 8, 'head_dim': 128,
            'vocab_size': 151936, 'tie_word_embeddings': True}
SOURCE = 'https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/blob/main/config.json'


def estimate(c, architecture=None):
    a = architecture or OFFICIAL
    h, inner, layers = (a[k] for k in ('hidden_size', 'intermediate_size', 'num_hidden_layers'))
    q, kv = a['num_attention_heads'] * a['head_dim'], a['num_key_value_heads'] * a['head_dim']
    vocab = a['vocab_size']
    dimensions = {'q_proj': (h, q), 'k_proj': (h, kv), 'v_proj': (h, kv), 'o_proj': (q, h),
                  'gate_proj': (h, inner), 'up_proj': (h, inner), 'down_proj': (inner, h)}
    matrices = layers * sum(x*y for x, y in dimensions.values())
    embedding = h*vocab * (1 if a['tie_word_embeddings'] else 2)
    norms = layers * (2*h + 2*a['head_dim']) + h
    parameters = matrices + embedding + norms
    lora = layers * c['lora']['r'] * sum(sum(dimensions[k]) for k in c['lora']['target_modules'])
    context, active, micro = c['max_context'], c.get('rollout_batch_size', 1), c.get('logprob_batch_size', 1)
    groups, attempts, replicas = c.get('grpo_groups_per_update', 1), c.get('grpo_max_group_attempts', 1), c['group_size']
    steps, turns = c['grpo_steps'], c['max_searches'] + 1
    # NF4 values + approximate nested quantization metadata; tied embeddings/norms kept FP32 by PEFT.
    base = matrices * .52 + (embedding + norms) * 4
    kv_elements = 2 * layers * kv * active * context
    every = c.get('grpo_checkpoint_every', 1)
    snapshots = (steps + every - 1) // every
    checkpoint = lora * 12  # adapter FP32 + Adam first and second moments; gradients not serialized
    return {
        'kind': 'planning_estimate_not_measurement', 'architecture_source': SOURCE,
        'architecture': a, 'parameter_count': parameters, 'lora_parameters': lora,
        'config': {k: c.get(k) for k in ('max_context', 'max_action_tokens', 'max_searches', 'group_size',
                    'grpo_groups_per_update', 'grpo_max_group_attempts', 'rollout_batch_size', 'logprob_batch_size')},
        'trajectories': {'target_per_update': groups*replicas, 'maximum_per_round': attempts*replicas,
                         'maximum_per_100_rounds': attempts*replicas*100},
        'gpu_gib': {
            'nf4_base_with_fp32_nonquantized_approx': base/GIB,
            'policy_reference_grad_adam_fp32': lora*20/GIB,
            'kv_cache_bf16_at_full_context': kv_elements*2/GIB,
            'kv_cache_fp32_at_full_context': kv_elements*4/GIB,
            'scoring_full_logits_fp32': micro*context*vocab*4/GIB,
            'action_softmax_single_copy_fp32': micro*c['max_action_tokens']*vocab*4/GIB,
            'checkpoint_layer_inputs_fp32': layers*micro*context*h*4/GIB,
            'note': 'Components have different lifetimes; not a peak sum. KV dtype and activation/workspace peaks require CUDA measurement.',
        },
        'disk_gib': {
            'original_bf16_weights': parameters*2/GIB,
            'one_adapter_fp32': lora*4/GIB,
            'one_checkpoint_weights_adam': checkpoint/GIB,
            'checkpoint_count': snapshots,
            'all_checkpoints_weights_adam': checkpoint*snapshots/GIB,
            'tokenizer_metadata_allowance_all_checkpoints': snapshots*16*1024**2/GIB,
            'note': 'Snapshot frequency follows grpo_checkpoint_every; final round is always saved. Excludes previous runs, environment, raw data, logs and temporary write headroom.',
        },
        'host_ram_gib': {
            'max_admitted_prefix_and_action_python_ints': attempts*replicas*turns*context*36/GIB,
            'selected_old_reference_logps_fp32_upper': groups*replicas*turns*c['max_action_tokens']*8/GIB,
            'note': 'CPython int+list slot modeled at36 bytes. Excludes over-budget rejected prompts, messages, dictionaries, BM25, libraries, model-loading buffers and a prior round still referenced.',
        },
        'assumptions': ['LoRA all listed linear targets, bias=none, FP32 adapters and Adam moments',
                        'One NF4 base shared by policy/reference, no full second reference model',
                        'Generation and backward are sequential; generation16 and scoring2 for the selected profile',
                        'All scheduled snapshots retained; compressed rollout traces enabled by selected profile'],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--output', default='reports/search/resource-estimate.json')
    a = p.parse_args()
    c = read_json(a.config)
    local_model = path(c['model_path']) / 'config.json'
    report = estimate(c, read_json(local_model) if local_model.is_file() else None)
    report['architecture_from_local_model'] = local_model.is_file()
    write_json(a.output, report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
