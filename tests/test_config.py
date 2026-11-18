from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
SEED = 42


def test_config_has_reproducible_training_defaults() -> None:
    with CONFIG_PATH.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    assert config["project"]["seed"] == SEED
    assert config["training"]["seed"] == SEED
    assert config["training"]["lookback_steps"] > 0
    assert config["training"]["forecast_horizon"] > 0


def test_config_paths_are_relative() -> None:
    with CONFIG_PATH.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    paths = [
        *config["paths"]["data"].values(),
        *config["paths"]["models"].values(),
        *config["paths"]["reports"].values(),
    ]

    assert all(not Path(path).is_absolute() for path in paths)
