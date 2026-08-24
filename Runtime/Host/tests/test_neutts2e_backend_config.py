from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from Runtime.Voice.neutts2e_backend import NeuTTS2EBackend


class NeuTTS2EBackendConfigTests(unittest.TestCase):
    def test_production_defaults_use_arc_and_pipelined_perth(self) -> None:
        keys = {
            "DEXTER_TTS_VARIANT": "",
            "DEXTER_TTS_BACKBONE_DEVICE": "",
            "DEXTER_TTS_VULKAN_DEVICE": "",
            "DEXTER_TTS_PIPELINE_WATERMARK": "",
            "GGML_VK_VISIBLE_DEVICES": "",
        }
        clean = {key: value for key, value in os.environ.items() if key not in keys}
        with patch.dict(os.environ, clean, clear=True):
            env = NeuTTS2EBackend.runtime_env()

        self.assertEqual(env["DEXTER_TTS_VARIANT"], "q8")
        self.assertEqual(env["DEXTER_TTS_BACKBONE_DEVICE"], "gpu")
        self.assertEqual(env["DEXTER_TTS_VULKAN_DEVICE"], "0")
        self.assertEqual(env["DEXTER_TTS_PIPELINE_WATERMARK"], "1")
        self.assertEqual(env["GGML_VK_VISIBLE_DEVICES"], "0")

    def test_explicit_device_override_is_preserved(self) -> None:
        with patch.dict(
            os.environ,
            {
                "DEXTER_TTS_BACKBONE_DEVICE": "cpu",
                "DEXTER_TTS_PIPELINE_WATERMARK": "0",
            },
            clear=False,
        ):
            env = NeuTTS2EBackend.runtime_env()

        self.assertEqual(env["DEXTER_TTS_BACKBONE_DEVICE"], "cpu")
        self.assertEqual(env["DEXTER_TTS_PIPELINE_WATERMARK"], "0")


if __name__ == "__main__":
    unittest.main()
