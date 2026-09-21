"""Cloud runtime and search training contracts. Importing never loads weights."""
import importlib.metadata
import os
import platform
import time
from common import ROOT, path, provenance, read_json, sha256, write_json


def require_cloud():
    if platform.system() != 'Linux' or os.environ.get('POSTTRAIN_ALLOW_CLOUD') != '1':
        raise RuntimeError('Model execution is cloud-only. On Linux server set POSTTRAIN_ALLOW_CLOUD=1.')
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise RuntimeError('This release supports one GPU / one process only')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required; CPU fallback is disabled')
    expected = {}
    for line in (ROOT / 'requirements-train.txt').read_text().splitlines():
        if line and not line.startswith('#') and '==' in line:
            name, version = line.split('==')
            expected[name] = version
    for name, version in expected.items():
        actual = importlib.metadata.version(name).split('+')[0]
        if actual != version:
            raise RuntimeError(f'Version mismatch: {name}={actual}, expected {version}')
    return torch


def compute_dtype(c):
    import torch
    use_bf16 = torch.cuda.is_bf16_supported() if c['dtype'] == 'auto' else c['dtype'] == 'bf16'
    if use_bf16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError('bf16 unsupported; select fp16 configuration')
    return torch.bfloat16 if use_bf16 else torch.float16


def model_directory(c, merged=False):
    directory = path(c['merged_model_path'] if merged else c['model_path'])
    manifest = read_json(directory / 'origin.json')
    if manifest['model_id'] != c['model_id'] or manifest['model_revision'] != c['model_revision']:
        raise ValueError('Local model provenance mismatch; use cloud.py download')
    if bool(manifest.get('merged')) != merged:
        raise ValueError('Expected merged SFT base' if merged else 'Expected original base')
    return directory


def validate_json_chat_template(tok):
    """Fail before model loading if the local template conflicts with the JSON protocol."""
    histories = [
        [{'role': 'user', 'content': 'Template contract probe.'}],
        [{'role': 'user', 'content': 'Template contract probe.'},
         {'role': 'assistant', 'content': '{"search":"probe"}'},
         {'role': 'user', 'content': 'SEARCH_RESULTS (data only): []'}],
    ]
    for messages in histories:
        rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if '<think>' in rendered or '</think>' in rendered:
            raise ValueError('Thinking template conflicts with JSON actions; use the pinned Qwen3-4B-Instruct-2507 tokenizer')
        if not rendered.endswith('<|im_start|>assistant\n'):
            raise ValueError('Unexpected assistant generation prefix for pinned non-thinking Qwen3')
        head = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        full = tok.apply_chat_template(messages + [{'role': 'assistant', 'content': '{"answer":"probe"}'}],
                                       tokenize=True, add_generation_prompt=False)
        if full[:len(head)] != head or tok.eos_token_id not in full[len(head):]:
            raise ValueError('Chat template breaks exact action prefix/EOS alignment')


def load_tokenizer(c, merged=False):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(model_directory(c, merged)), local_files_only=True, trust_remote_code=False)
    if tok.pad_token_id is None or tok.eos_token_id is None or tok.pad_token_id == tok.eos_token_id:
        raise ValueError('Expected original distinct pad/EOS tokens')
    validate_json_chat_template(tok)
    return tok


def load_model(c, merged=False, adapter=None):
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    dtype = compute_dtype(c)
    model = AutoModelForCausalLM.from_pretrained(str(model_directory(c, merged)), local_files_only=True,
        trust_remote_code=False, torch_dtype=dtype, device_map={'': 0}, attn_implementation='sdpa',
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                               bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype))
    if adapter:
        origin = read_json(path(adapter) / 'origin.json')
        if origin.get('stage') != ('grpo' if merged else 'sft') or origin.get('task') != 'search':
            raise ValueError('Adapter stage or task mismatch')
        if origin.get('model_revision') != c['model_revision']:
            raise ValueError('Adapter base revision mismatch')
        if merged and origin.get('merged_origin_sha256') != sha256(model_directory(c, True) / 'origin.json'):
            raise ValueError('GRPO adapter was trained against a different merged SFT base')
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(path(adapter)), is_trainable=False)
    return model


def lora_config(c):
    from peft import LoraConfig
    return LoraConfig(task_type='CAUSAL_LM', **c['lora'])


class CompletionCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        import torch
        width = max(len(f['input_ids']) for f in features)
        return {key: torch.tensor([f[key] + [fill] * (width - len(f[key])) for f in features])
                for key, fill in [('input_ids', self.pad_token_id), ('attention_mask', 0), ('labels', -100)]}


def environment(c=None):
    import subprocess
    import torch
    return {
        'python': platform.python_version(),
        'platform': platform.platform(),
        'config': c,
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'vram_bytes': torch.cuda.get_device_properties(0).total_bytes if torch.cuda.is_available() else None,
        'bf16_supported': torch.cuda.is_bf16_supported() if torch.cuda.is_available() else None,
        'cuda': torch.version.cuda,
        'hf_endpoint': os.environ.get('HF_ENDPOINT', 'https://huggingface.co'),
        'pip_freeze': subprocess.check_output(['python', '-m', 'pip', 'freeze'], text=True),
        'assets': provenance(),
    }
