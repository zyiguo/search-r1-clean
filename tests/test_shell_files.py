import unittest
from common import ROOT


class ShellFileTests(unittest.TestCase):
    def test_scripts_are_linux_text(self):
        for file in (ROOT / 'scripts').glob('*.sh'):
            with self.subTest(file=file.name):
                data = file.read_bytes()
                self.assertNotIn(b'\r', data)
                self.assertFalse(data.startswith(b'\xef\xbb\xbf'))
                self.assertTrue(data.startswith(b'#!/usr/bin/env bash\n'))
                data.decode('utf-8')
