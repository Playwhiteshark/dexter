from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import Runtime.Profiles.loader as loader
from Runtime.Profiles.loader import ProfileLoadError, ProfileValidationError


class ProfileLoaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.patch_root = patch.object(loader, "PROFILES_ROOT", self.root)
        self.patch_root.start()
        self.addCleanup(self.patch_root.stop)

    def _manifest(self, **overrides):
        value = {
            "schema_version": 1,
            "id": "dexter",
            "name": "Dexter",
            "model": {
                "provider": "ollama",
                "model": "gemma4-jarvis:12b",
            },
            "instructions": "instructions.md",
            "voice": {
                "backend": "neutts2e",
                "preset": "production",
                "speaker": "paul",
                "emotion": "neutral",
            },
            "tools": {"preset": "default"},
            "appearance": {"accent": "default"},
        }
        value.update(overrides)
        return value

    def _write_profile(
        self,
        *,
        profile_id: str = "dexter",
        manifest=None,
        instructions: str = "Be concise.",
    ) -> Path:
        profile_dir = self.root / profile_id
        profile_dir.mkdir(parents=True)
        if manifest is None:
            manifest = self._manifest(id=profile_id)
        (profile_dir / "profile.json").write_text(
            json.dumps(manifest),
            encoding="utf-8",
        )
        (profile_dir / "instructions.md").write_text(
            instructions,
            encoding="utf-8",
        )
        return profile_dir

    def test_loads_normalized_immutable_profile(self) -> None:
        self._write_profile(instructions="Use tools carefully.\n")

        profile = loader.load_profile("dexter")

        self.assertEqual(profile.id, "dexter")
        self.assertEqual(profile.name, "Dexter")
        self.assertEqual(profile.model.provider, "ollama")
        self.assertEqual(profile.model.model, "gemma4-jarvis:12b")
        self.assertEqual(profile.instructions, "Use tools carefully.\n")
        self.assertEqual(profile.voice.backend, "neutts2e")
        self.assertEqual(profile.voice.preset, "production")
        self.assertEqual(profile.voice.speaker, "paul")
        self.assertEqual(profile.voice.emotion, "neutral")
        self.assertEqual(profile.tools.preset, "default")
        self.assertEqual(profile.appearance.accent, "default")
        with self.assertRaises(AttributeError):
            profile.name = "Changed"  # type: ignore[misc]

    def test_voice_speaker_and_emotion_can_be_omitted_for_other_backends(self) -> None:
        manifest = self._manifest(
            voice={
                "backend": "reference-voice",
                "preset": "casual",
            }
        )
        self._write_profile(manifest=manifest)

        profile = loader.load_profile("dexter")

        self.assertEqual(profile.voice.backend, "reference-voice")
        self.assertEqual(profile.voice.preset, "casual")
        self.assertIsNone(profile.voice.speaker)
        self.assertIsNone(profile.voice.emotion)

    def test_empty_instruction_file_is_valid(self) -> None:
        self._write_profile(instructions="")
        self.assertEqual(loader.load_profile("dexter").instructions, "")

    def test_profile_id_rejects_traversal_and_path_syntax(self) -> None:
        for value in ("", ".", "..", "../Host", r"..\Host", "a/b", "-dexter"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                loader.load_profile(value)

    def test_profile_id_allows_version_like_names(self) -> None:
        self._write_profile(profile_id="dexter-casual-2.0")
        profile = loader.load_profile("dexter-casual-2.0")
        self.assertEqual(profile.id, "dexter-casual-2.0")

    def test_missing_profile_is_file_not_found(self) -> None:
        with self.assertRaisesRegex(FileNotFoundError, "Profile not found"):
            loader.load_profile("missing")

    def test_missing_manifest_is_file_not_found(self) -> None:
        (self.root / "dexter").mkdir()
        with self.assertRaisesRegex(FileNotFoundError, "profile.json"):
            loader.load_profile("dexter")

    def test_invalid_json_preserves_profile_context(self) -> None:
        profile_dir = self.root / "dexter"
        profile_dir.mkdir()
        (profile_dir / "profile.json").write_text("{ nope", encoding="utf-8")

        with self.assertRaisesRegex(ProfileLoadError, "Invalid JSON.*dexter"):
            loader.load_profile("dexter")

    def test_schema_version_must_match(self) -> None:
        self._write_profile(manifest=self._manifest(schema_version=2))
        with self.assertRaisesRegex(ProfileValidationError, "schema_version"):
            loader.load_profile("dexter")

    def test_manifest_id_must_match_requested_profile(self) -> None:
        self._write_profile(manifest=self._manifest(id="casual"))
        with self.assertRaisesRegex(ProfileValidationError, "Profile ID mismatch"):
            loader.load_profile("dexter")

    def test_unknown_fields_are_rejected_instead_of_silently_ignored(self) -> None:
        self._write_profile(manifest=self._manifest(modle={"oops": True}))
        with self.assertRaisesRegex(ProfileValidationError, "Unknown field.*modle"):
            loader.load_profile("dexter")

    def test_manifest_resource_cannot_escape_profile_directory(self) -> None:
        outside = self.root / "outside.md"
        outside.write_text("secret", encoding="utf-8")
        self._write_profile(manifest=self._manifest(instructions="../outside.md"))

        with self.assertRaisesRegex(ProfileValidationError, "escapes profile directory"):
            loader.load_profile("dexter")

    def test_nested_profile_resources_are_allowed(self) -> None:
        manifest = self._manifest(instructions="prompts/system.md")
        profile_dir = self._write_profile(manifest=manifest)
        prompts = profile_dir / "prompts"
        prompts.mkdir()
        (prompts / "system.md").write_text("nested", encoding="utf-8")

        self.assertEqual(loader.load_profile("dexter").instructions, "nested")

    def test_lists_only_valid_profile_directories_with_manifests(self) -> None:
        self._write_profile(profile_id="dexter")
        self._write_profile(profile_id="casual-2.0")
        (self.root / "not a profile").mkdir()
        (self.root / "missing-manifest").mkdir()
        (self.root / "README.md").write_text("x", encoding="utf-8")

        self.assertEqual(loader.list_profile_ids(), ("casual-2.0", "dexter"))


if __name__ == "__main__":
    unittest.main()
