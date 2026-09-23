from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from game.config import (
    CONFIG_PATH_ENV,
    ENV_FILE_ENV,
    describe_sources,
    find_config_file,
    find_env_file,
    resolve_model_env,
)


class ConfigResolutionTest(unittest.TestCase):
    """模型接入配置的解析顺序与来源发现。

    原实现只读 Path.cwd()/.env，换目录启动就读不到配置。这些测试钉住
    "在任意目录下都能取到 URL 和 key"，以及三种配置方式的优先级。
    """

    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.root = Path(self._dir.name)
        self.addCleanup(self._dir.cleanup)
        # 屏蔽真实环境，避免本机 .env 干扰断言。
        self._clean_env = {
            ENV_FILE_ENV: os.devnull,
            CONFIG_PATH_ENV: os.devnull,
        }

    def _write_env(self, **values: str) -> Path:
        path = self.root / ".env"
        path.write_text(
            "\n".join(f"{k}={v}" for k, v in values.items()), encoding="utf-8"
        )
        return path

    def _write_toml(self, body: str, name: str = "werewolf.toml") -> Path:
        path = self.root / name
        path.write_text(body, encoding="utf-8")
        return path

    def test_env_file_is_found_upward_not_from_cwd(self) -> None:
        # 关键回归：从子目录启动也要能找到上层的 .env。
        self._write_env(WEREWOLF_MODEL_API_KEY="from-dotenv")
        nested = self.root / "a" / "b" / "c"
        nested.mkdir(parents=True)
        with mock.patch.dict(os.environ, {}, clear=True):
            found = find_env_file(nested)
        self.assertIsNotNone(found)
        self.assertEqual(Path(found).parent, self.root)

    def test_explicit_env_file_wins(self) -> None:
        custom = self.root / "custom.env"
        custom.write_text("WEREWOLF_MODEL_API_KEY=explicit", encoding="utf-8")
        with mock.patch.dict(os.environ, {ENV_FILE_ENV: str(custom)}, clear=True):
            self.assertEqual(find_env_file(self.root), custom)
            values = resolve_model_env(start=self.root)
        self.assertEqual(values["WEREWOLF_MODEL_API_KEY"], "explicit")

    def test_toml_config_supplies_url_and_key(self) -> None:
        self._write_toml(
            '[model]\nbase_url = "https://toml.example"\napi_key = "toml-key"\n'
            'task_model = "toml-model"\n'
        )
        with mock.patch.dict(os.environ, {ENV_FILE_ENV: os.devnull}, clear=True):
            values = resolve_model_env(start=self.root)
        self.assertEqual(values["WEREWOLF_MODEL_BASE_URL"], "https://toml.example")
        self.assertEqual(values["WEREWOLF_MODEL_API_KEY"], "toml-key")
        self.assertEqual(values["WEREWOLF_TASK_MODEL"], "toml-model")

    def test_process_env_beats_config_file(self) -> None:
        self._write_toml('[model]\napi_key = "from-toml"\n')
        env = dict(self._clean_env)
        env[CONFIG_PATH_ENV] = str(self.root / "werewolf.toml")
        env["WEREWOLF_MODEL_API_KEY"] = "from-process"
        with mock.patch.dict(os.environ, env, clear=True):
            values = resolve_model_env(start=self.root)
        self.assertEqual(values["WEREWOLF_MODEL_API_KEY"], "from-process")

    def test_config_file_beats_dotenv(self) -> None:
        self._write_env(WEREWOLF_MODEL_API_KEY="from-dotenv")
        toml_path = self._write_toml('[model]\napi_key = "from-toml"\n')
        env = {ENV_FILE_ENV: str(self.root / ".env"), CONFIG_PATH_ENV: str(toml_path)}
        with mock.patch.dict(os.environ, env, clear=True):
            values = resolve_model_env(start=self.root)
        self.assertEqual(values["WEREWOLF_MODEL_API_KEY"], "from-toml")

    def test_overrides_beat_everything(self) -> None:
        self._write_env(WEREWOLF_MODEL_API_KEY="from-dotenv")
        env = dict(self._clean_env)
        env["WEREWOLF_MODEL_API_KEY"] = "from-process"
        with mock.patch.dict(os.environ, env, clear=True):
            values = resolve_model_env(
                start=self.root, overrides={"WEREWOLF_MODEL_API_KEY": "explicit"}
            )
        self.assertEqual(values["WEREWOLF_MODEL_API_KEY"], "explicit")

    def test_legacy_names_still_work(self) -> None:
        # 已有 .env 用的是 OPENAI_* 名字，不能因为改名而失效。
        env = dict(self._clean_env)
        env["OPENAI_API_KEY"] = "legacy-key"
        env["OPENAI_BASE_URL"] = "https://legacy.example"
        with mock.patch.dict(os.environ, env, clear=True):
            values = resolve_model_env(start=self.root)
        self.assertEqual(values["WEREWOLF_MODEL_API_KEY"], "legacy-key")
        self.assertEqual(values["WEREWOLF_MODEL_BASE_URL"], "https://legacy.example")

    def test_new_names_backfill_legacy_readers(self) -> None:
        # 只配新名时，沿用旧名的读取路径也要能取到。
        env = dict(self._clean_env)
        env["WEREWOLF_MODEL_API_KEY"] = "new-key"
        with mock.patch.dict(os.environ, env, clear=True):
            values = resolve_model_env(start=self.root)
        self.assertEqual(values["OPENAI_API_KEY"], "new-key")

    def test_missing_config_is_not_an_error(self) -> None:
        with mock.patch.dict(os.environ, self._clean_env, clear=True):
            self.assertIsNone(find_config_file(self.root))
            values = resolve_model_env(start=self.root)
        self.assertIsInstance(values, dict)

    def test_malformed_toml_reports_path(self) -> None:
        bad = self._write_toml("[model\nbroken")
        with mock.patch.dict(os.environ, {CONFIG_PATH_ENV: str(bad)}, clear=True):
            with self.assertRaises(ValueError) as caught:
                resolve_model_env(start=self.root)
        self.assertIn("werewolf.toml", str(caught.exception))

    def test_describe_sources_reports_actual_files(self) -> None:
        env_path = self._write_env(WEREWOLF_MODEL_API_KEY="k")
        with mock.patch.dict(os.environ, {ENV_FILE_ENV: str(env_path)}, clear=True):
            sources = describe_sources(self.root)
        self.assertEqual(sources["env_file"], str(env_path))

    def test_example_config_is_shipped(self) -> None:
        example = Path(__file__).resolve().parents[1] / "werewolf.toml.example"
        self.assertTrue(example.is_file())
        text = example.read_text(encoding="utf-8")
        self.assertIn("base_url", text)
        self.assertIn("api_key", text)
        # 示例不能包含看起来像真实密钥的内容。
        self.assertIn("replace-with-your-key", text)


if __name__ == "__main__":
    unittest.main()
