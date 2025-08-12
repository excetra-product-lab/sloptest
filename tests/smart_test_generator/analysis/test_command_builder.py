from pathlib import Path
from smart_test_generator.analysis.coverage.command_builder import build_pytest_command
from smart_test_generator.config import Config


def test_build_pytest_command_defaults_uses_python_module(tmp_path: Path):
    cfg = Config()
    spec = build_pytest_command(project_root=tmp_path, config=cfg)
    # Should use current interpreter -m pytest style
    assert '-m' in spec.argv and 'pytest' in spec.argv
    assert str(tmp_path) in ' '.join(spec.argv)
    assert spec.cwd == tmp_path
    assert isinstance(spec.env, dict)


def test_build_pytest_command_pytest_path(tmp_path: Path):
    cfg = Config()
    cfg.config['test_generation']['coverage']['runner']['mode'] = 'pytest-path'
    cfg.config['test_generation']['coverage']['runner']['pytest_path'] = 'pytest'
    spec = build_pytest_command(project_root=tmp_path, config=cfg)
    assert spec.argv[0] == 'pytest'
    assert spec.cwd == tmp_path

