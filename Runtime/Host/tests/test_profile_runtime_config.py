from __future__ import annotations

import importlib.util
import os
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from Runtime.Host.acp_process import _agent_environment
from Runtime.Host.agent_profile import ModelProfile, load_profile
from Runtime.Host.gateway_process import GatewayProcess


_GATEWAY_PATH = (
    Path(__file__).resolve().parents[2]
    / "Model Gateway"
    / "gateway.py"
)


class ProfileRuntimeConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = replace(
            load_profile(),
            model=ModelProfile(
                provider="test-provider",
                name="test-model:1b",
                reasoning_effort="low",
            ),
        )

    def test_goose_environment_comes_from_profile(self) -> None:
        env = _agent_environment(self.profile)

        self.assertEqual(env["GOOSE_PROVIDER"], "test-provider")
        self.assertEqual(env["GOOSE_MODEL"], "test-model:1b")

    def test_gateway_child_environment_comes_from_profile(self) -> None:
        process = GatewayProcess(_GATEWAY_PATH, profile=self.profile)

        with patch.dict(
            os.environ,
            {
                "DEXTER_MODEL": "wrong-model",
                "DEXTER_REASONING_EFFORT": "wrong-effort",
            },
        ):
            env = process._gateway_environment()

        self.assertEqual(env["DEXTER_MODEL"], "test-model:1b")
        self.assertEqual(env["DEXTER_REASONING_EFFORT"], "low")

    def test_gateway_script_reads_child_environment(self) -> None:
        with patch.dict(
            os.environ,
            {
                "DEXTER_MODEL": "test-model:1b",
                "DEXTER_REASONING_EFFORT": "low",
            },
        ):
            spec = importlib.util.spec_from_file_location(
                "dexter_profile_gateway_probe",
                _GATEWAY_PATH,
            )
            self.assertIsNotNone(spec)
            self.assertIsNotNone(spec.loader)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

        self.assertEqual(module.PRODUCTION_MODEL, "test-model:1b")
        self.assertEqual(module.DEFAULT_REASONING_EFFORT, "low")


if __name__ == "__main__":
    unittest.main()
