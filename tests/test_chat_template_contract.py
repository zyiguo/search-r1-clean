import unittest
from training_common import validate_json_chat_template
from search_task import action


class Tokenizer:
    eos_token_id = 9
    def __init__(self, thinking=False, mismatch=False):
        self.thinking, self.mismatch = thinking, mismatch
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        if not tokenize:
            return '<|im_start|>assistant\n' + ('<think>\n' if self.thinking else '')
        if add_generation_prompt:
            return [1, 2, 3]
        return [1, 2, 4 if self.mismatch else 3, 8, 9]


class TemplateTests(unittest.TestCase):
    def test_non_thinking_prefix_is_accepted(self):
        validate_json_chat_template(Tokenizer())

    def test_thinking_template_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Thinking template'):
            validate_json_chat_template(Tokenizer(thinking=True))

    def test_rewritten_prefix_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'alignment'):
            validate_json_chat_template(Tokenizer(mismatch=True))

    def test_reasoning_tags_are_not_silently_stripped(self):
        with self.assertRaises(ValueError):
            action('<think>text</think>{"answer":"yes"}')
