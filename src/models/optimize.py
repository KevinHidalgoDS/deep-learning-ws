# -*- coding: utf-8 -*-  # noqa: UP009
"""Búsqueda de hiperparámetros con Optuna y exportación del modelo campeón (Sprint 3).

Para cada arquitectura de ``config.models.architectures`` se ejecuta un estudio de Optuna
(muestreador TPE con semilla y poda por mediana) sobre:

- tasa de aprendizaje (escala logarítmica),
- tamaño de ventana (``lookback_steps``): se recortan por la izquierda las ventanas ya guardadas,
  así todas las pruebas se evalúan sobre los mismos objetivos,
- unidades ocultas (``hidden_units``),
- *dropout*.

La métrica de cada prueba y de la selección final es el **MAE de validación en COP/kWh**. La
partición de prueba no se lee: queda intacta para el Sprint 4.

Salidas:

- ``models/checkpoints/optuna_<arquitectura>_trials.csv``: todas las pruebas.
- ``models/checkpoints/optuna_<arquitectura>_best.pt``: pesos de la mejor prueba.
- ``models/trained/best_params.json``: mejores hiperparámetros por arquitectura y ranking.
- ``models/trained/champion.pt`` (red) o ``models/trained/champion.pkl`` (si gana Ridge), y
  ``models/trained/champion.json`` con sus metadatos.

Ejemplo:
    $ python -m src.models.optimize
    $ python -m src.models.optimize --models tcn --trials 10 --epochs 15

Proyecto: deep-learning-ws

Programa: optimize.py

version: 1.0.0

lenguaje: Python 3.14.5

CD: 20261124

LUD: 20261124

Comentarios:
    - 2026-11-24 Sebastian Sanchez -> creación.
"""

__authors__ = ["Sebastian Sanchez"]
__copyright__ = "Copyright 2026, Universidad Nacional de Colombia"
__credits__ = ["Sebastian Sanchez"]
__status__ = "Desarrollo"
__version__ = "1.0.0"
__date__ = "2026-11-24"

import argparse
import json
import logging
import shutil
from collections.abc import Callable
from pathlib import Path

import joblib
import numpy as np
import optuna
import yaml

from src.config import DEFAULT_CONFIG_PATH, PROJECT_PATH, read_config_mapping
from src.models.architectures import point_forecast
from src.models.train_model import (
    BASELINE_FILE,
    Datasets,
    TrainSettings,
    load_datasets,
    load_network,
    network_meta,
    predict,
    save_network,
    seasonal_naive_metrics,
    train_baseline,
    train_network,
)
from src.utils.logger import log_event

# -----------------------------------------------------------------------------
# CONFIGURACIÓN DE CONSTANTES Y LOGGING DEL MÓDULO
# -----------------------------------------------------------------------------
BEST_PARAMS_FILE = "best_params.json"
CHAMPION_META_FILE = "champion.json"
CHAMPION_NETWORK_FILE = "champion.pt"
CHAMPION_RIDGE_FILE = "champion.pkl"
PRUNER_WARMUP_EPOCHS = 5

logger = logging.getLogger(__name__)
if not logger.handlers:
    _console_handler = logging.StreamHandler()
    _console_handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
    )
    logger.addHandler(_console_handler)
    logger.setLevel(logging.INFO)


# -----------------------------------------------------------------------------
# ESPACIO DE BÚSQUEDA Y OBJETIVO
# -----------------------------------------------------------------------------
def lookback_choices(space: dict, max_lookback: int) -> list[int]:
    """Tamaños de ventana válidos: los del espacio que no superan la ventana guardada.

    Raises:
        ValueError: Si ninguno cabe en la ventana de los tensores.
    """
    choices = sorted(v for v in space["lookback_steps"] if v <= max_lookback)
    if not choices:
        raise ValueError(
            f"Ningún lookback de {space['lookback_steps']} cabe en {max_lookback} pasos"
        )
    return choices


def suggest(
    trial: optuna.Trial, base_params: dict, space: dict, max_lookback: int
) -> tuple[dict, float, int]:
    """Muestrea hiperparámetros; los no buscados conservan su valor de la configuración.

    Returns:
        Tupla ``(params de la arquitectura, tasa de aprendizaje, lookback)``.
    """
    low, high = space["learning_rate"]
    learning_rate = trial.suggest_float("learning_rate", low, high, log=True)
    lookback = trial.suggest_categorical("lookback_steps", lookback_choices(space, max_lookback))
    params = dict(base_params)
    params["hidden_units"] = trial.suggest_categorical("hidden_units", list(space["hidden_units"]))
    params["dropout"] = trial.suggest_float("dropout", *space["dropout"])
    return params, learning_rate, lookback


def make_objective(
    name: str, data: Datasets, config: dict, checkpoint: Path
) -> Callable[[optuna.Trial], float]:
    """Crea la función objetivo de Optuna para una arquitectura.

    Cada prueba entrena con early stopping y devuelve el MAE de validación en COP/kWh. La pérdida
    de validación de cada época se reporta a Optuna para podar pruebas poco prometedoras. Los
    pesos de la mejor prueba se guardan en ``checkpoint`` en cuanto aparecen, así no hay que
    reentrenar (ni depender de que el reentrenamiento sea idéntico) para exportarla.
    """
    optimization = config["optimization"]
    space = optimization["search_space"]
    max_lookback = data.train.x.shape[1]
    best = {"mae": float("inf")}

    def objective(trial: optuna.Trial) -> float:
        params, learning_rate, lookback = suggest(
            trial, config["models"][name], space, max_lookback
        )
        settings = TrainSettings.from_config(
            config["training"], learning_rate=learning_rate, epochs=optimization["epochs"]
        )

        def report(epoch: int, val_loss: float) -> None:
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        result, summary = train_network(
            name, params, data.cropped(lookback), config, settings=settings, on_epoch=report
        )
        mae = summary["val_cop"]["mae"]
        trial.set_user_attr("val_rmse_cop", summary["val_cop"]["rmse"])
        trial.set_user_attr("best_epoch", summary["best_epoch"])
        trial.set_user_attr("parameters", summary["parameters"])
        if mae < best["mae"]:
            best["mae"] = mae
            save_network(
                checkpoint,
                result.model,
                network_meta(summary, data, config) | {"trial": trial.number},
            )
        return mae

    return objective


def run_study(name: str, data: Datasets, config: dict, *, n_trials: int, checkpoints: Path) -> dict:
    """Ejecuta el estudio de una arquitectura y resume su mejor prueba."""
    optimization = config["optimization"]
    seed = config["training"]["seed"]
    study = optuna.create_study(
        study_name=f"{config['project']['name']}-{name}",
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=PRUNER_WARMUP_EPOCHS),
    )
    checkpoint = checkpoints / f"optuna_{name}_best.pt"
    timeout = optimization.get("timeout_minutes")
    study.optimize(
        make_objective(name, data, config, checkpoint),
        n_trials=n_trials,
        timeout=timeout * 60 if timeout else None,
        gc_after_trial=True,
    )
    study.trials_dataframe().to_csv(checkpoints / f"optuna_{name}_trials.csv", index=False)
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not complete:
        raise ValueError(f"Ninguna prueba de {name} terminó; aumenta n_trials o epochs")
    best = study.best_trial
    return {
        "architecture": name,
        "best_trial": best.number,
        "trials": {
            "complete": len(complete),
            "pruned": sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials),
        },
        "params": best.params,
        "val_cop": {"mae": best.value, "rmse": best.user_attrs["val_rmse_cop"]},
        "best_epoch": best.user_attrs["best_epoch"],
        "parameters": best.user_attrs["parameters"],
        "checkpoint": str(checkpoint),
    }


# -----------------------------------------------------------------------------
# CAMPEÓN
# -----------------------------------------------------------------------------
def export_champion(ranking: list[dict], trained: Path, baseline_path: Path) -> dict:
    """Copia el mejor modelo (menor MAE de validación) a ``models/trained/``.

    Returns:
        Metadatos del campeón, también escritos en ``champion.json``.
    """
    champion = ranking[0]
    trained.mkdir(parents=True, exist_ok=True)
    if champion["architecture"] == "ridge":
        target = trained / CHAMPION_RIDGE_FILE
        shutil.copyfile(baseline_path, target)
        kind = "sklearn"
    else:
        target = trained / CHAMPION_NETWORK_FILE
        shutil.copyfile(champion["checkpoint"], target)
        kind = "torch"
    meta = {
        "architecture": champion["architecture"],
        "kind": kind,
        "path": str(target),
        "params": champion["params"],
        "lookback": champion.get("lookback") or champion["params"].get("lookback_steps"),
        "selection_metric": "val_cop.mae",
        "val_cop": champion["val_cop"],
    }
    (trained / CHAMPION_META_FILE).write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return meta


def load_champion(trained: Path) -> tuple[Callable[[np.ndarray], np.ndarray], dict]:
    """Carga el campeón exportado como una función de predicción homogénea.

    La función recibe ventanas ``(B, L_guardada, F)`` del Sprint 2, las recorta al ``lookback``
    del campeón y devuelve el pronóstico puntual ``(B, H)`` en escala del objetivo (aplicar
    ``scaler.pkl`` para pasarlo a COP/kWh).
    """
    meta = json.loads((trained / CHAMPION_META_FILE).read_text(encoding="utf-8"))
    if meta["kind"] == "sklearn":
        model = joblib.load(meta["path"])
        return (lambda x: model.predict(x[:, -meta["lookback"] :])[..., 0]), meta
    network, net_meta = load_network(Path(meta["path"]))
    models_config = {"loss": net_meta["loss"], "quantiles": net_meta["quantiles"]}

    def forecast(x: np.ndarray) -> np.ndarray:
        return point_forecast(
            predict(network, x[:, -net_meta["lookback"] :], device="cpu"), models_config
        )

    return forecast, meta | {"network": net_meta}


def run(
    config: dict,
    *,
    architectures: list[str] | None = None,
    n_trials: int | None = None,
    epochs: int | None = None,
    root: Path = PROJECT_PATH,
) -> dict:
    """Optimiza cada arquitectura, la compara contra los baselines y exporta el campeón."""
    if epochs is not None:
        config = {**config, "optimization": {**config["optimization"], "epochs": epochs}}
    data = load_datasets(config, root=root)
    paths = config["paths"]["models"]
    checkpoints, trained = root / paths["checkpoints"], root / paths["trained"]
    checkpoints.mkdir(parents=True, exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    baseline, baseline_summary = train_baseline(data, config["models"]["baseline"]["ridge_alpha"])
    baseline_path = checkpoints / BASELINE_FILE
    joblib.dump(baseline, baseline_path)
    reference = seasonal_naive_metrics(data)

    trials = n_trials or config["optimization"]["n_trials"]
    ranking = [baseline_summary]
    for name in architectures or config["models"]["architectures"]:
        logger.info("Optimizando %s (%d pruebas)", name, trials)
        summary = run_study(name, data, config, n_trials=trials, checkpoints=checkpoints)
        logger.info(
            "%s: mejor MAE val %.2f COP/kWh con %s",
            name,
            summary["val_cop"]["mae"],
            summary["params"],
        )
        ranking.append(summary)

    ranking.sort(key=lambda item: item["val_cop"]["mae"])
    champion = export_champion(ranking, trained, baseline_path)
    report = {
        "selection_metric": "val_cop.mae",
        "validation_samples": len(data.validation.x),
        "reference_seasonal_naive": reference["val_cop"] if reference else None,
        "ranking": ranking,
        "champion": champion,
    }
    (trained / BEST_PARAMS_FILE).write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    log_event(logging.INFO, "optimization_finished", champion=champion["architecture"])
    return report


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada de línea de comandos de la optimización."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--models", nargs="+", help="Subconjunto de arquitecturas a optimizar")
    parser.add_argument("--trials", type=int, help="Reemplaza optimization.n_trials")
    parser.add_argument("--epochs", type=int, help="Reemplaza optimization.epochs")
    args = parser.parse_args(argv)
    try:
        report = run(
            read_config_mapping(args.config),
            architectures=args.models,
            n_trials=args.trials,
            epochs=args.epochs,
        )
    except (ValueError, OSError, KeyError, yaml.YAMLError) as exc:
        logger.exception("Fallo en la optimización: %s", exc)
        log_event(logging.ERROR, "optimization_failed", error=str(exc))
        return 1
    for item in report["ranking"]:
        logger.info("%-16s MAE %8.2f COP/kWh", item["architecture"], item["val_cop"]["mae"])
    if report["reference_seasonal_naive"]:
        logger.info(
            "Persistencia 24 h: MAE %.2f COP/kWh", report["reference_seasonal_naive"]["mae"]
        )
    logger.info("Campeón: %s", report["champion"]["architecture"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
