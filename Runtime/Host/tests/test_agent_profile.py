from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from Runtime.Host.agent_profile import ProfileError, load_profile


class AgentProfileTests(unittest.TestCase):
    def test_loads_current_dexter_profile(self) -> None:
        profile = load_profile()

        self.assertEqual(profile.profile_id, "dexter")
        self.assertEqual(profile.display_name, "Dexter")
        self.assertEqual(profile.model.provider, "ollama")
        self.assertEqual(profile.model.name, "gemma4-jarvis:12b")
        self.assertEqual(profile.model.reasoning_effort, "none")
        self.assertEqual(profile.voice.backend, "neutts2e")
        self.assertEqual(profile.voice.preset, "production_arc")
        self.assertEqual(profile.voice.speaker, "paul")
        self.assertEqual(profile.voice.emotion, "neutral")
        self.assertEqual(profile.tools.preset, "dexter_default")
        self.assertIsNone(profile.appearance.accent)
        self.assertIsNone(profile.instructions_file)

    def test_rejects_profile_id_path_traversal(self) -> None:
        with self.assertRaisesRegex(ProfileError, "profile_id may contain"):
            load_profile("../dexter")

    def test_rejects_unknown_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            profile_dir = root / "test"
            profile_dir.mkdir()
            payload = self._valid_payload()
            payload["modle"] = payload["model"]
            (profile_dir / "profile.json").write_text(
                json.dumps(payload),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ProfileError, "unknown fields"):
                load_profile("test", profiles_dir=root)

    def test_instructions_file_cannot_escape_profile_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            profile_dir = root / "test"
            profile_dir.mkdir()
            (root / "outside.md").write_text("outside", encoding="utf-8")
            payload = self._valid_payload()
            payload["instructions_file"] = "../outside.md"
            (profile_dir / "profile.json").write_text(
                json.dumps(payload),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ProfileError, "must stay inside"):
                load_profile("test", profiles_dir=root)

    @staticmethod
    def _valid_payload() -> dict[str, object]:
        return {
            "schema_version": 1,
            "id": "test",
            "display_name": "Test",
            "instructions_file": None,
            "model": {
                "provider": "ollama",
                "name": "test-model",
                "reasoning_effort": "none",
            },
            "voice": {
                "backend": "neutts2e",
                "preset": "production_arc",
                "speaker": "paul",
                "emotion": "neutral",
            },
            "tools": {"preset": "dexter_default"},
            "appearance": {"accent": None},
        }


if __name__ == "__main__":
    unittest.main()
