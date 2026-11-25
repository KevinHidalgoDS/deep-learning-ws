# -*- coding: utf-8 -*-  # noqa: UP009
"""Arquitecturas para pronosticar el precio de bolsa a varias horas (Sprint 3).

Todas las redes reciben ventanas ``(muestras, pasos_tiempo, características)`` generadas por
``src.features.build_features`` y devuelven ``(muestras, horizonte, salidas)``. ``salidas`` vale 1
con la pérdida de Huber (pronóstico puntual) o el número de cuantiles con la pérdida cuantílica
(intervalos de predicción). El pronóstico es directo y multisalida: las 24 h se predicen a la vez,
sin realimentar predicciones, para que el error no se acumule a lo largo del horizonte.

Modelos:

- ``RidgeBaseline``: regresión lineal regularizada sobre la ventana aplanada. Es el baseline de
  ML que toda red debe superar para justificar su complejidad.
- ``seasonal_naive``: persistencia estacional (el precio de mañana a la misma hora es el de hoy).
  Se reporta como referencia mínima.
- ``LSTMAttention``: red recurrente con atención aditiva sobre los pasos de la ventana.
- ``TCN``: red convolucional temporal con convoluciones causales dilatadas.
- ``NBeats``: perceptrón profundo con bloques residuales de backcast/forecast.

Las razones de cada arquitectura frente a los datos están en el docstring de su clase.

Ejemplo:
    >>> model = build_model("tcn", n_features=35, lookback=168, horizon=24, params={})
    >>> model(torch.zeros(8, 168, 35)).shape
    torch.Size([8, 24, 1])

Proyecto: deep-learning-ws

Programa: architectures.py

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

from collections.abc import Sequence
from typing import Self

import numpy as np
import torch
from sklearn.linear_model import Ridge
from torch import nn
from torch.nn.utils.parametrizations import weight_norm

ARCHITECTURES = ("lstm_attention", "tcn", "nbeats")
LOSSES = ("huber", "quantile")
MEDIAN = 0.5


# -----------------------------------------------------------------------------
# FUNCIONES DE PÉRDIDA
# -----------------------------------------------------------------------------
class HuberLoss(nn.Module):
    """Pérdida de Huber sobre el pronóstico puntual.

    Es cuadrática para errores pequeños y lineal para los grandes: los picos de escasez del
    precio de bolsa no dominan el gradiente como lo harían con MSE, pero el ajuste fino en
    condiciones normales sigue siendo suave. ``delta`` está en unidades escaladas del objetivo.
    """

    def __init__(self, delta: float = 1.0) -> None:
        super().__init__()
        self.loss = nn.HuberLoss(delta=delta)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compara ``prediction`` ``(B, H, 1)`` con ``target`` ``(B, H)``."""
        return self.loss(prediction[..., 0], target)


class QuantileLoss(nn.Module):
    """Pérdida pinball promediada sobre varios cuantiles.

    Penaliza de forma asimétrica: para el cuantil ``q`` subestimar cuesta ``q`` y sobrestimar
    ``1 - q``. Entrenar los cuantiles 0,1, 0,5 y 0,9 a la vez entrega un intervalo del 80 %.
    """

    def __init__(self, quantiles: Sequence[float]) -> None:
        super().__init__()
        validate_quantiles(quantiles)
        self.register_buffer("quantiles", torch.tensor(list(quantiles), dtype=torch.float32))

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compara ``prediction`` ``(B, H, Q)`` con ``target`` ``(B, H)``."""
        error = target.unsqueeze(-1) - prediction
        return torch.maximum(self.quantiles * error, (self.quantiles - 1) * error).mean()


def validate_quantiles(quantiles: Sequence[float]) -> None:
    """Exige cuantiles ordenados en ``(0, 1)`` que incluyan la mediana.

    Raises:
        ValueError: Si la lista no cumple las condiciones.
    """
    values = list(quantiles)
    if not values or any(not 0 < q < 1 for q in values) or values != sorted(set(values)):
        raise ValueError(f"Cuantiles inválidos {values}: deben ser crecientes y estar en (0, 1)")
    if MEDIAN not in values:
        raise ValueError("Los cuantiles deben incluir 0.5 para el pronóstico puntual")


def build_loss(models_config: dict) -> nn.Module:
    """Crea la pérdida definida en la sección ``models`` de la configuración.

    Raises:
        ValueError: Si la pérdida no está soportada.
    """
    kind = models_config.get("loss", "huber")
    if kind == "huber":
        return HuberLoss(models_config.get("huber_delta", 1.0))
    if kind == "quantile":
        return QuantileLoss(models_config["quantiles"])
    raise ValueError(f"Pérdida no soportada: {kind}; usa {LOSSES}")


def output_size(models_config: dict) -> int:
    """Número de salidas por paso del horizonte: 1 (Huber) o el número de cuantiles."""
    return len(models_config["quantiles"]) if models_config.get("loss") == "quantile" else 1


def point_forecast(prediction: np.ndarray, models_config: dict) -> np.ndarray:
    """Extrae el pronóstico puntual ``(B, H)``: la salida única o el cuantil 0,5."""
    if models_config.get("loss") == "quantile":
        return prediction[..., list(models_config["quantiles"]).index(MEDIAN)]
    return prediction[..., 0]


# -----------------------------------------------------------------------------
# BASELINES
# -----------------------------------------------------------------------------
class RidgeBaseline:
    """Regresión Ridge multisalida sobre la ventana aplanada.

    Con pocas muestras (unos meses de datos horarios) un modelo lineal bien regularizado suele
    ser difícil de batir: aprovecha los rezagos y las medias móviles casi tan bien como una red.
    Por eso es la vara con la que se mide si las redes aportan algo.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self.model = Ridge(alpha=alpha)

    def fit(self, x: np.ndarray, y: np.ndarray) -> Self:
        """Ajusta con ``x`` ``(B, L, F)`` y ``y`` ``(B, H)``."""
        self.model.fit(x.reshape(len(x), -1), y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Devuelve ``(B, H, 1)`` para que tenga la misma forma que las redes."""
        return self.model.predict(x.reshape(len(x), -1))[..., None].astype(np.float32)


def seasonal_naive(target_history: np.ndarray, horizon: int, season: int = 24) -> np.ndarray:
    """Persistencia estacional: repite el último ciclo observado del objetivo.

    Args:
        target_history: Objetivo en la ventana de entrada, forma ``(B, L)`` con ``L >= season``.
        horizon: Pasos a pronosticar.
        season: Periodo estacional en pasos (24 h por el ciclo diario del precio).

    Returns:
        Pronóstico ``(B, horizon)``: el paso ``t + h`` toma el valor de ``t + h - season``
        (o de ``t + h - k·season`` si ``h > season``).
    """
    if target_history.shape[1] < season:
        raise ValueError(
            f"La ventana ({target_history.shape[1]}) es menor que la estación ({season})"
        )
    last_cycle = target_history[:, -season:]
    return np.stack([last_cycle[:, h % season] for h in range(horizon)], axis=1)


# -----------------------------------------------------------------------------
# REDES PROFUNDAS
# -----------------------------------------------------------------------------
class ForecastHead(nn.Module):
    """Capa final común: normaliza, aplica dropout y proyecta a ``(B, H, salidas)``."""

    def __init__(self, in_features: int, horizon: int, n_outputs: int, dropout: float) -> None:
        super().__init__()
        self.horizon, self.n_outputs = horizon, n_outputs
        self.net = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(dropout),
            nn.Linear(in_features, horizon * n_outputs),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Proyecta ``(B, in_features)`` al horizonte."""
        return self.net(features).view(-1, self.horizon, self.n_outputs)


class LSTMAttention(nn.Module):
    """LSTM apilada con atención aditiva (Bahdanau) sobre la ventana de entrada.

    Por qué encaja: el precio depende de su historia reciente (ACF(1) = 0,86) pero también de lo
    que ocurrió a la misma hora 24 y 168 h atrás. Una LSTM sola resume la ventana en su último
    estado y tiende a olvidar los pasos lejanos; la atención le permite consultar directamente
    cualquier hora de la semana anterior. Los pesos de atención además son interpretables.
    """

    def __init__(  # noqa: PLR0913
        self,
        n_features: int,
        horizon: int,
        *,
        hidden_units: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        n_outputs: int = 1,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            n_features,
            hidden_units,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_units)
        self.query = nn.Linear(hidden_units, hidden_units, bias=False)
        self.key = nn.Linear(hidden_units, hidden_units)
        self.score = nn.Linear(hidden_units, 1, bias=False)
        self.head = ForecastHead(2 * hidden_units, horizon, n_outputs, dropout)
        self.last_attention: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` ``(B, L, F)`` → ``(B, H, salidas)``."""
        states, _ = self.lstm(x)
        states = self.norm(states)
        last = states[:, -1]
        energy = self.score(torch.tanh(self.key(states) + self.query(last).unsqueeze(1)))
        weights = torch.softmax(energy, dim=1)  # (B, L, 1): peso de cada hora de la ventana.
        self.last_attention = weights.detach().squeeze(-1)
        context = (weights * states).sum(dim=1)
        return self.head(torch.cat([context, last], dim=-1))


class TemporalBlock(nn.Module):
    """Bloque residual de dos convoluciones causales dilatadas."""

    def __init__(
        self, channels_in: int, channels_out: int, kernel_size: int, dilation: int, dropout: float
    ) -> None:
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv1 = weight_norm(
            nn.Conv1d(channels_in, channels_out, kernel_size, dilation=dilation)
        )
        self.conv2 = weight_norm(
            nn.Conv1d(channels_out, channels_out, kernel_size, dilation=dilation)
        )
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.residual = (
            nn.Conv1d(channels_in, channels_out, 1)
            if channels_in != channels_out
            else nn.Identity()
        )

    def causal_conv(self, conv: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Rellena solo a la izquierda: la salida en ``t`` usa entradas ``<= t``."""
        return conv(nn.functional.pad(x, (self.padding, 0)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` ``(B, C_in, L)`` → ``(B, C_out, L)``."""
        out = self.dropout(self.activation(self.causal_conv(self.conv1, x)))
        out = self.dropout(self.activation(self.causal_conv(self.conv2, out)))
        return self.activation(out + self.residual(x))


class TCN(nn.Module):
    """Temporal Convolutional Network (Bai et al., 2018).

    Por qué encaja: las dilataciones 1, 2, 4, …, 32 cubren con pocas capas toda la ventana
    semanal (campo receptivo de ``1 + 2·(k-1)·(2^niveles - 1)`` pasos) y capturan a la vez el
    patrón intradiario y el semanal. A diferencia de la LSTM, procesa la ventana en paralelo, por
    lo que entrena más rápido en CPU y no sufre gradientes que se desvanecen. Usa
    ``weight_norm`` en las convoluciones, como en el artículo original.
    """

    def __init__(  # noqa: PLR0913
        self,
        n_features: int,
        horizon: int,
        *,
        hidden_units: int = 32,
        levels: int = 6,
        kernel_size: int = 3,
        dropout: float = 0.2,
        n_outputs: int = 1,
    ) -> None:
        super().__init__()
        blocks = [
            TemporalBlock(
                n_features if i == 0 else hidden_units, hidden_units, kernel_size, 2**i, dropout
            )
            for i in range(levels)
        ]
        self.network = nn.Sequential(*blocks)
        self.receptive_field = 1 + 2 * (kernel_size - 1) * (2**levels - 1)
        self.head = ForecastHead(hidden_units, horizon, n_outputs, dropout)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Representación causal ``(B, C, L)`` de la ventana ``(B, L, F)``."""
        return self.network(x.transpose(1, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` ``(B, L, F)`` → ``(B, H, salidas)`` a partir del último paso codificado."""
        return self.head(self.encode(x)[:, :, -1])


class NBeatsBlock(nn.Module):
    """Bloque genérico de N-BEATS: MLP que emite un backcast y un forecast."""

    def __init__(
        self, input_size: int, horizon_size: int, hidden: int, layers: int, dropout: float
    ):
        super().__init__()
        stack: list[nn.Module] = []
        for i in range(layers):
            stack += [
                nn.Linear(input_size if i == 0 else hidden, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
        self.mlp = nn.Sequential(*stack)
        self.backcast = nn.Linear(hidden, input_size)
        self.forecast = nn.Linear(hidden, horizon_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Devuelve ``(backcast, forecast)``."""
        hidden = self.mlp(x)
        return self.backcast(hidden), self.forecast(hidden)


class NBeats(nn.Module):
    """N-BEATS genérico (Oreshkin et al., 2020) con entrada multivariada aplanada.

    Por qué encaja: no supone recurrencia ni convolución; cada bloque explica una parte de la
    ventana (backcast), la resta y deja el residuo al siguiente, que se concentra en lo que aún
    no se explicó (por ejemplo, el pico de la tarde tras modelar el nivel). Con pocas muestras es
    una buena prueba de si la estructura secuencial de LSTM/TCN realmente aporta. La versión
    original es univariada; aquí la ventana se aplana para incluir las variables exógenas, al
    estilo de N-BEATSx, y se normaliza la entrada con ``LayerNorm``.
    """

    def __init__(  # noqa: PLR0913
        self,
        n_features: int,
        lookback: int,
        horizon: int,
        *,
        hidden_units: int = 256,
        num_blocks: int = 3,
        layers_per_block: int = 4,
        dropout: float = 0.1,
        n_outputs: int = 1,
    ) -> None:
        super().__init__()
        self.horizon, self.n_outputs = horizon, n_outputs
        input_size = lookback * n_features
        self.norm = nn.LayerNorm(input_size)
        self.blocks = nn.ModuleList(
            NBeatsBlock(input_size, horizon * n_outputs, hidden_units, layers_per_block, dropout)
            for _ in range(num_blocks)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` ``(B, L, F)`` → ``(B, H, salidas)``: suma de los forecasts de cada bloque."""
        residual = self.norm(x.flatten(1))
        forecast = 0
        for block in self.blocks:
            backcast, block_forecast = block(residual)
            residual = residual - backcast
            forecast = forecast + block_forecast
        return forecast.view(-1, self.horizon, self.n_outputs)


def build_model(  # noqa: PLR0913
    name: str,
    *,
    n_features: int,
    lookback: int,
    horizon: int,
    params: dict,
    n_outputs: int = 1,
) -> nn.Module:
    """Crea una red por nombre con los hiperparámetros de ``config.models.<name>``.

    Args:
        name: ``lstm_attention``, ``tcn`` o ``nbeats``.
        n_features: Características por paso.
        lookback: Pasos de la ventana de entrada.
        horizon: Pasos a pronosticar.
        params: Hiperparámetros de la arquitectura; las claves ausentes toman su valor por defecto.
        n_outputs: Salidas por paso (1 o el número de cuantiles).

    Raises:
        ValueError: Si la arquitectura no existe.
    """
    if name == "lstm_attention":
        return LSTMAttention(n_features, horizon, n_outputs=n_outputs, **params)
    if name == "tcn":
        return TCN(n_features, horizon, n_outputs=n_outputs, **params)
    if name == "nbeats":
        return NBeats(n_features, lookback, horizon, n_outputs=n_outputs, **params)
    raise ValueError(f"Arquitectura no soportada: {name}; usa {ARCHITECTURES}")


def count_parameters(model: nn.Module) -> int:
    """Número de parámetros entrenables."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
