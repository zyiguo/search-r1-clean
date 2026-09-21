"""Run the existing trainer with five console fields; preserve full recorded metrics."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def grpo_console(text):
    if not isinstance(text, str):
        return text
    try:
        values = json.loads(text)
    except (ValueError, TypeError):
        return text
    if not isinstance(values, dict) or not {'step','loss','signal_updates'} <= values.keys():
        return text
    return json.dumps({key:values.get(key) for key in
                      ('step','loss','reward','effective_group_fraction','kl_per_token')})


class SFTConsole:
    def __init__(self):
        self.values = {'loss':None,'last_val_loss':None,'learning_rate':None,'grad_norm':None}

    def update(self, step, logs):
        for key in self.values:
            source = 'eval_loss' if key == 'last_val_loss' else key
            if source in logs:
                self.values[key] = logs[source]
        return {'step':step, **self.values}


def main():
    # Patch only console callbacks and the trainer module's print, never metric inputs.
    import builtins
    import search_train as train
    if '--help' in sys.argv or '-h' in sys.argv:
        return train.main()
    from transformers.trainer_callback import ProgressCallback, PrinterCallback
    from tqdm.auto import tqdm
    console = SFTConsole()
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero:
            tqdm.write(json.dumps(console.update(state.global_step, logs or {})))
    def compact_print(*args, **kwargs):
        if len(args) == 1:
            args = (grpo_console(args[0]),)
        builtins.print(*args, **kwargs)
    callbacks = (ProgressCallback, PrinterCallback)
    originals = [cls.on_log for cls in callbacks]
    had_print = 'print' in vars(train)
    original_print = getattr(train, 'print', None)
    try:
        for cls in callbacks:
            cls.on_log = on_log
        train.print = compact_print
        train.main()
    finally:
        for cls, original in zip(callbacks, originals):
            cls.on_log = original
        if had_print:
            train.print = original_print
        else:
            del train.print


if __name__ == '__main__':
    main()
