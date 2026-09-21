import tempfile
import unittest
from pathlib import Path
from hardware_probe import available_ram, profile_errors


class HardwareTests(unittest.TestCase):
    def test_container_memory_limit_overrides_host(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'meminfo').write_text('MemAvailable: 100000000 kB\n')
            (root/'memory.max').write_text(str(40*1024**3))
            (root/'memory.current').write_text(str(12*1024**3))
            self.assertEqual(available_ram(root, root), 28*1024**3)

    def test_profile_accepts_expected_resources_and_rejects_small_disk(self):
        gib = 1024**3
        self.assertEqual(profile_errors(84*gib, 80*gib, 100*gib, (12,0), '12.8', True), [])
        errors = profile_errors(24*gib, 16*gib, 20*gib, (12,0), '12.6', False)
        self.assertEqual(len(errors), 5)


if __name__ == '__main__': unittest.main()
