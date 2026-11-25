# -*- coding: utf-8 -*-  # noqa: UP009
"""Entrenamiento de las redes y del baseline sobre los tensores del Sprint 2 (Sprint 3).

Lee ``train.npz`` y ``validation.npz`` de ``data/processed/`` y el escalador de
``models/trained/scaler.pkl``. El conjunto de prueba no se abre aquí: se reserva para la
evaluación fuera de muestra del Sprint 4.

Para cada arquitectura de ``config.models.architectures``:

1. Entrena con AdamW, recorte de gradiente y ``ReduceLROnPlateau`` sobre la pérdida de
   validación.
2. Aplica *early stopping* sobre la pérdida de validación y guarda los mejores pesos en
   ``models/checkpoints/<arquitectura>_best.pt``.
3. Registra pérdida, MAE y RMSE de entrenamiento y validación por época en
   ``models/checkpoints/<arquitectura>_history.csv``.

Además ajusta el baseline Ridge y calcula la persistencia estacional. El resumen comparativo en
COP/kWh sobre validación queda en ``models/checkpoints/training_summary.json``.

Todos los hiperparámetros salen de ``config/config.yaml`` (secciones ``training`` y ``models``).

Ejemplo:
    $ python -m src.models.train_model
    $ python -m src.models.train_model --models tcn nbeats --epochs 20

Proyecto: deep-learning-ws

Programa: train_model.py

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
import copy
import csv
import json
import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import DEFAULT_CONFIG_PATH, PROJECT_PATH, read_config_mapping
from src.features.build_features import METADATA_FILE
from src.models.architectures import (
    RidgeBaseline,
    build_loss,
    build_model,
    count_parameters,
    output_size,
    point_forecast,
    seasonal_naive,
)
from src.utils.logger import log_event

# -----------------------------------------------------------------------------
# CONFIGURACIÓN DE CONSTANTES Y LOGGING DEL MÓDULO
# -----------------------------------------------------------------------------
FIT_SPLITS = ("train", "validation")
SUMMARY_FILE = "training_summary.json"
BASELINE_FILE = "ridge_baseline.pkl"
SEASON_HOURS = 24

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
# DATOS
# -----------------------------------------------------------------------------
@dataclass
class Split:
    """Ventanas de una partición: ``x`` ``(B, L, F)``, ``y`` ``(B, H)`` e instantes de corte."""

    x: np.ndarray
    y: np.ndarray
    input_end: np.ndarray

    def cropped(self, lookback: int) -> Self:
        """Conserva solo los últimos ``lookback`` pasos de cada ventana.

        Recortar por la izquierda mantiene la causalidad (la ventana sigue terminando en ``t``) y
        las mismas muestras, de modo que ventanas de distinto tamaño se comparan sobre los
        mismos objetivos.
        """
        if not 0 < lookback <= self.x.shape[1]:
            raise ValueError(f"lookback {lookback} fuera de (0, {self.x.shape[1]}]")
        return type(self)(self.x[:, -lookback:], self.y, self.input_end)


@dataclass
class Datasets:
    """Particiones de ajuste, metadatos de características y escalador del Sprint 2."""

    train: Split
    validation: Split
    metadata: dict
    scalers: dict

    @property
    def target_index(self) -> int:
        """Posición del precio (escalado como característica) en el eje de características."""
        return self.metadata["feature_columns"].index(self.metadata["target"])

    def cropped(self, lookback: int) -> Self:
        """Copia con ambas particiones recortadas al ``lookback`` dado."""
        return type(self)(
            self.train.cropped(lookback),
            self.validation.cropped(lookback),
            self.metadata,
            self.scalers,
        )


def load_split(processed: Path, name: str) -> Split:
    """Lee ``<name>.npz`` de ``data/processed/``.

    Raises:
        FileNotFoundError: Si el archivo no existe.
    """
    path = processed / f"{name}.npz"
    if not path.is_file():
        raise FileNotFoundError(
            f"No existe {path}; ejecuta primero `python -m src.features.build_features`."
        )
    with np.load(path) as data:
        return Split(data["X"], data["y"], data["input_end"])


def check_chronology(train: Split, validation: Split) -> None:
    """Exige que toda ventana de validación termine después de la última de entrenamiento.

    Raises:
        ValueError: Si hay solapamiento temporal entre particiones.
    """
    last_train = pd.to_datetime(train.input_end, utc=True).max()
    first_val = pd.to_datetime(validation.input_end, utc=True).min()
    if first_val <= last_train:
        raise ValueError(f"Validación ({first_val}) no es posterior a entrenamiento ({last_train})")


def load_datasets(config: dict, *, root: Path = PROJECT_PATH) -> Datasets:
    """Carga entrenamiento, validación, metadatos y escalador; nunca la partición de prueba."""
    paths = config["paths"]
    processed = root / paths["data"]["processed"]
    metadata = json.loads((processed / METADATA_FILE).read_text(encoding="utf-8"))
    scalers = joblib.load(root / paths["models"]["trained"] / config["features"]["scaler_file"])
    train, validation = (load_split(processed, name) for name in FIT_SPLITS)
    check_chronology(train, validation)
    if train.x.shape[1:] != validation.x.shape[1:]:
        raise ValueError(f"Formas incompatibles: {train.x.shape} vs {validation.x.shape}")
    return Datasets(train, validation, metadata, scalers)


def to_cop(values: np.ndarray, scalers: dict) -> np.ndarray:
    """Invierte el escalado del objetivo para expresar ``values`` en COP/kWh."""
    flat = np.asarray(values, dtype=np.float64).reshape(-1, 1)
    return scalers["target"].inverse_transform(flat).reshape(np.shape(values))


def feature_target_to_cop(values: np.ndarray, scalers: dict, target: str) -> np.ndarray:
    """Pasa el precio escalado como característica (no como objetivo) a COP/kWh."""
    scaler = scalers["features"]
    i = list(scalers["feature_columns"]).index(target)
    center = getattr(scaler, "center_", getattr(scaler, "mean_", None))
    offset = 0.0 if center is None else center[i]
    return np.asarray(values, dtype=np.float64) * scaler.scale_[i] + offset


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """MAE y RMSE sobre todas las horas del horizonte, en las unidades de entrada."""
    error = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
    return {"mae": float(np.mean(np.abs(error))), "rmse": float(np.sqrt(np.mean(error**2)))}


# -----------------------------------------------------------------------------
# ENTRENAMIENTO
# -----------------------------------------------------------------------------
@dataclass
class TrainSettings:
    """Parámetros del ciclo de entrenamiento, leídos de ``config.training``."""

    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    patience: int
    min_delta: float = 0.0
    gradient_clip_norm: float | None = 1.0
    scheduler_factor: float = 0.5
    scheduler_patience: int = 4
    min_lr: float = 1e-5
    seed: int = 42
    device: str = "auto"
    num_workers: int = 0

    @classmethod
    def from_config(cls, training: dict, **overrides: object) -> Self:
        """Construye la configuración; ``overrides`` reemplaza valores (p. ej. desde Optuna)."""
        scheduler = training.get("lr_scheduler", {})
        values = {
            "epochs": training["epochs"],
            "batch_size": training["batch_size"],
            "learning_rate": training["learning_rate"],
            "weight_decay": training["weight_decay"],
            "patience": training["early_stopping_patience"],
            "min_delta": training.get("early_stopping_min_delta", 0.0),
            "gradient_clip_norm": training.get("gradient_clip_norm", 1.0),
            "scheduler_factor": scheduler.get("factor", 0.5),
            "scheduler_patience": scheduler.get("patience", 4),
            "min_lr": scheduler.get("min_lr", 1e-5),
            "seed": training.get("seed", 42),
            "device": training.get("device", "auto"),
            "num_workers": training.get("num_workers", 0),
        }
        values.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**values)


@dataclass
class TrainResult:
    """Resultado de entrenar una red: mejores pesos y registro por época."""

    model: nn.Module
    best_epoch: int
    best_val_loss: float
    history: list[dict] = field(default_factory=list)
    stopped_early: bool = False


def set_seed(seed: int) -> None:
    """Fija las semillas de Python, NumPy y PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(name: str) -> torch.device:
    """``auto`` usa CUDA si está disponible y CPU en otro caso."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def make_loader(
    split: Split, batch_size: int, *, shuffle: bool, seed: int, workers: int
) -> DataLoader:
    """DataLoader reproducible. Barajar el entrenamiento no filtra el futuro: cada muestra ya es
    una ventana completa cuyo objetivo es posterior a su entrada."""
    dataset = TensorDataset(torch.from_numpy(split.x).float(), torch.from_numpy(split.y).float())
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, generator=generator, num_workers=workers
    )


@torch.no_grad()
def predict(
    model: nn.Module, x: np.ndarray, *, batch_size: int = 256, device: str = "auto"
) -> np.ndarray:
    """Predice ``(B, H, salidas)`` en modo evaluación (sin dropout)."""
    target_device = resolve_device(device)
    model.to(target_device).eval()
    outputs = [
        model(torch.from_numpy(x[i : i + batch_size]).float().to(target_device)).cpu().numpy()
        for i in range(0, len(x), batch_size)
    ]
    return np.concatenate(outputs) if outputs else np.empty((0,), dtype=np.float32)


def _run_epoch(  # noqa: PLR0913
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    *,
    device: torch.device,
    models_config: dict,
    optimizer: torch.optim.Optimizer | None = None,
    clip: float | None = None,
) -> dict[str, float]:
    """Una pasada por ``loader``; entrena si recibe ``optimizer``. Métricas en escala del modelo."""
    training = optimizer is not None
    model.train(training)
    total_loss, abs_error, sq_error, count, n_values = 0.0, 0.0, 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for batch_x, batch_y in loader:
            x, y = batch_x.to(device), batch_y.to(device)
            prediction = model(x)
            loss = loss_fn(prediction, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                if clip:
                    nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
            point = torch.from_numpy(
                point_forecast(prediction.detach().cpu().numpy(), models_config)
            )
            error = point - y.cpu()
            total_loss += loss.item() * len(x)
            abs_error += error.abs().sum().item()
            sq_error += (error**2).sum().item()
            count += len(x)
            n_values += y.numel()
    return {
        "loss": total_loss / count,
        "mae": abs_error / n_values,
        "rmse": (sq_error / n_values) ** 0.5,
    }


def fit_network(  # noqa: PLR0913
    model: nn.Module,
    train: Split,
    validation: Split,
    *,
    loss_fn: nn.Module,
    settings: TrainSettings,
    models_config: dict,
    on_epoch: Callable[[int, float], None] | None = None,
) -> TrainResult:
    """Entrena con early stopping sobre la pérdida de validación.

    Args:
        model: Red sin entrenar.
        train: Partición de entrenamiento.
        validation: Partición de validación, posterior a la de entrenamiento.
        loss_fn: Pérdida (Huber o cuantílica).
        settings: Parámetros del ciclo.
        models_config: Sección ``models`` (para extraer el pronóstico puntual).
        on_epoch: Función opcional ``(época, pérdida_validación)``; Optuna la usa para podar
            pruebas poco prometedoras lanzando una excepción.

    Returns:
        ``TrainResult`` con el modelo cargado con los mejores pesos.
    """
    device = resolve_device(settings.device)
    model.to(device)
    loss_fn.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=settings.scheduler_factor,
        patience=settings.scheduler_patience,
        min_lr=settings.min_lr,
    )
    loaders = {
        "train": make_loader(
            train,
            settings.batch_size,
            shuffle=True,
            seed=settings.seed,
            workers=settings.num_workers,
        ),
        "validation": make_loader(
            validation,
            settings.batch_size,
            shuffle=False,
            seed=settings.seed,
            workers=settings.num_workers,
        ),
    }

    best_loss, best_epoch, best_state, wait = float("inf"), 0, copy.deepcopy(model.state_dict()), 0
    history: list[dict] = []
    stopped_early = False
    for epoch in range(1, settings.epochs + 1):
        start = time.perf_counter()
        train_metrics = _run_epoch(
            model,
            loaders["train"],
            loss_fn,
            device=device,
            models_config=models_config,
            optimizer=optimizer,
            clip=settings.gradient_clip_norm,
        )
        val_metrics = _run_epoch(
            model, loaders["validation"], loss_fn, device=device, models_config=models_config
        )
        scheduler.step(val_metrics["loss"])
        history.append(
            {
                "epoch": epoch,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()},
                "learning_rate": optimizer.param_groups[0]["lr"],
                "seconds": time.perf_counter() - start,
            }
        )
        if val_metrics["loss"] < best_loss - settings.min_delta:
            best_loss, best_epoch, wait = val_metrics["loss"], epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
        if on_epoch is not None:
            on_epoch(epoch, val_metrics["loss"])
        if wait >= settings.patience:
            stopped_early = True
            logger.info("Early stopping en la época %d (mejor: %d)", epoch, best_epoch)
            break

    model.load_state_dict(best_state)
    return TrainResult(model, best_epoch, best_loss, history, stopped_early)


# -----------------------------------------------------------------------------
# PERSISTENCIA
# -----------------------------------------------------------------------------
def save_network(path: Path, model: nn.Module, meta: dict) -> Path:
    """Guarda pesos y todo lo necesario para reconstruir la red sin la configuración."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({**meta, "state_dict": model.state_dict()}, path)
    return path


def load_network(path: Path, *, device: str = "cpu") -> tuple[nn.Module, dict]:
    """Reconstruye una red guardada con ``save_network``.

    Returns:
        Tupla ``(modelo en modo evaluación, metadatos)``.
    """
    bundle = torch.load(path, map_location=device, weights_only=True)
    meta = {k: v for k, v in bundle.items() if k != "state_dict"}
    model = build_model(
        meta["architecture"],
        n_features=meta["n_features"],
        lookback=meta["lookback"],
        horizon=meta["horizon"],
        params=meta["params"],
        n_outputs=meta["n_outputs"],
    )
    model.load_state_dict(bundle["state_dict"])
    return model.eval(), meta


def write_history(path: Path, history: list[dict]) -> Path:
    """Escribe el registro por época en CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    return path


# -----------------------------------------------------------------------------
# FLUJOS DE ALTO NIVEL
# -----------------------------------------------------------------------------
def validation_metrics_cop(prediction: np.ndarray, data: Datasets, models_config: dict) -> dict:
    """MAE y RMSE de validación en COP/kWh a partir de la salida del modelo."""
    point = point_forecast(prediction, models_config)
    return regression_metrics(to_cop(data.validation.y, data.scalers), to_cop(point, data.scalers))


def train_network(  # noqa: PLR0913
    name: str,
    params: dict,
    data: Datasets,
    config: dict,
    *,
    settings: TrainSettings | None = None,
    on_epoch: Callable[[int, float], None] | None = None,
) -> tuple[TrainResult, dict]:
    """Construye y entrena una arquitectura; devuelve el resultado y su resumen en validación."""
    settings = settings or TrainSettings.from_config(config["training"])
    models_config = config["models"]
    set_seed(settings.seed)
    lookback, n_features = data.train.x.shape[1:]
    horizon = data.train.y.shape[1]
    model = build_model(
        name,
        n_features=n_features,
        lookback=lookback,
        horizon=horizon,
        params=params,
        n_outputs=output_size(models_config),
    )
    result = fit_network(
        model,
        data.train,
        data.validation,
        loss_fn=build_loss(models_config),
        settings=settings,
        models_config=models_config,
        on_epoch=on_epoch,
    )
    prediction = predict(result.model, data.validation.x, device=settings.device)
    summary = {
        "architecture": name,
        "params": params,
        "learning_rate": settings.learning_rate,
        "lookback": int(lookback),
        "parameters": count_parameters(result.model),
        "best_epoch": result.best_epoch,
        "epochs_run": len(result.history),
        "stopped_early": result.stopped_early,
        "val_loss": result.best_val_loss,
        "val_cop": validation_metrics_cop(prediction, data, models_config),
    }
    return result, summary


def network_meta(summary: dict, data: Datasets, config: dict) -> dict:
    """Metadatos que acompañan a los pesos guardados."""
    models_config = config["models"]
    return {
        "architecture": summary["architecture"],
        "params": summary["params"],
        "n_features": int(data.train.x.shape[2]),
        "lookback": summary["lookback"],
        "horizon": int(data.train.y.shape[1]),
        "n_outputs": output_size(models_config),
        "loss": models_config.get("loss", "huber"),
        "quantiles": list(models_config.get("quantiles", [])),
        "feature_columns": list(data.metadata["feature_columns"]),
        "best_epoch": summary["best_epoch"],
        "val_loss": summary["val_loss"],
        "val_cop": summary["val_cop"],
    }


def train_baseline(data: Datasets, alpha: float) -> tuple[RidgeBaseline, dict]:
    """Ajusta Ridge con entrenamiento y lo evalúa en validación (COP/kWh)."""
    model = RidgeBaseline(alpha).fit(data.train.x, data.train.y)
    prediction = model.predict(data.validation.x)[..., 0]
    metrics = regression_metrics(
        to_cop(data.validation.y, data.scalers), to_cop(prediction, data.scalers)
    )
    return model, {
        "architecture": "ridge",
        "params": {"alpha": alpha},
        "lookback": int(data.train.x.shape[1]),
        "val_cop": metrics,
    }


def seasonal_naive_metrics(data: Datasets) -> dict | None:
    """Persistencia estacional de 24 h en validación (COP/kWh).

    Returns:
        Resumen de métricas, o ``None`` si la ventana es más corta que un día.
    """
    if data.validation.x.shape[1] < SEASON_HOURS:
        logger.warning("Ventana menor a %d h: se omite la persistencia estacional", SEASON_HOURS)
        return None
    target = data.metadata["target"]
    history = feature_target_to_cop(
        data.validation.x[:, :, data.target_index], data.scalers, target
    )
    forecast = seasonal_naive(history, data.validation.y.shape[1], SEASON_HOURS)
    return {
        "architecture": "seasonal_naive_24h",
        "val_cop": regression_metrics(to_cop(data.validation.y, data.scalers), forecast),
    }


def run(
    config: dict,
    *,
    architectures: list[str] | None = None,
    epochs: int | None = None,
    root: Path = PROJECT_PATH,
) -> dict:
    """Entrena baseline y redes con la configuración base y guarda checkpoints y resumen.

    Returns:
        Resumen con las métricas de validación de cada modelo, ordenado por MAE en COP/kWh.
    """
    data = load_datasets(config, root=root)
    checkpoints = root / config["paths"]["models"]["checkpoints"]
    checkpoints.mkdir(parents=True, exist_ok=True)
    settings = TrainSettings.from_config(config["training"], epochs=epochs)
    logger.info(
        "Entrenamiento: X%s validación X%s en %s",
        data.train.x.shape,
        data.validation.x.shape,
        resolve_device(settings.device),
    )

    baseline, baseline_summary = train_baseline(data, config["models"]["baseline"]["ridge_alpha"])
    joblib.dump(baseline, checkpoints / BASELINE_FILE)
    results = [baseline_summary]
    if (naive := seasonal_naive_metrics(data)) is not None:
        results.append(naive)

    for name in architectures or config["models"]["architectures"]:
        logger.info("Entrenando %s", name)
        result, summary = train_network(
            name, dict(config["models"][name]), data, config, settings=settings
        )
        summary["checkpoint"] = str(
            save_network(
                checkpoints / f"{name}_best.pt", result.model, network_meta(summary, data, config)
            )
        )
        summary["history"] = str(write_history(checkpoints / f"{name}_history.csv", result.history))
        results.append(summary)
        logger.info("%s: MAE val %.2f COP/kWh", name, summary["val_cop"]["mae"])

    results.sort(key=lambda item: item["val_cop"]["mae"])
    report = {
        "selection_metric": "val_cop.mae",
        "train_samples": len(data.train.x),
        "validation_samples": len(data.validation.x),
        "validation_period": [
            str(data.validation.input_end[0]),
            str(data.validation.input_end[-1]),
        ],
        "models": results,
    }
    (checkpoints / SUMMARY_FILE).write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    log_event(logging.INFO, "training_finished", ranking=[m["architecture"] for m in results])
    return report


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada de línea de comandos del entrenamiento."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--models", nargs="+", help="Subconjunto de arquitecturas a entrenar")
    parser.add_argument("--epochs", type=int, help="Reemplaza training.epochs")
    args = parser.parse_args(argv)
    try:
        report = run(
            read_config_mapping(args.config), architectures=args.models, epochs=args.epochs
        )
    except (ValueError, OSError, KeyError, yaml.YAMLError) as exc:
        logger.exception("Fallo en el entrenamiento: %s", exc)
        log_event(logging.ERROR, "training_failed", error=str(exc))
        return 1
    for item in report["models"]:
        logger.info(
            "%-20s MAE %8.2f  RMSE %8.2f COP/kWh",
            item["architecture"],
            item["val_cop"]["mae"],
            item["val_cop"]["rmse"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
