---
title: "Entregable 3: Fase de Definición (IHC)"
author: "_Milena Castaño_ & _Lissy Grace_ & _Kelly Monsalve_ & _Sebastian Sanchez_ & _Kevin Hidalgo_"
date: "14 de Septiembre de 2026"
institute: "Universidad Nacional de Colombia, Sede Medellín"
description:
  "lorem ipsum."

# --- Configuraciones para Pandoc (PDF) ---
lang: es-CO
bibliography: docs/referencias.bib
csl: docs/ieee.csl

# Formato visual del PDF (LaTeX)
geometry:
  - top=2.5cm
  - bottom=2.5cm
  - left=3cm
  - right=2.5cm
fontsize: 12pt
linestretch: 1.15
papersize: letter
toc: true
toc-depth: 2
---

# 🧠 Deep Learning - 3011087

## 📖 Descripción del Proyecto

Este repositorio establece la estructura estándar y las mejores prácticas para los proyectos,
prácticas y talleres del curso **Deep Learning** del Departamento de Ciencias de la Computación y
la Decisión de la **Universidad Nacional de Colombia, Sede Medellín**.

Repositorio oficial y guia del curso Deep Learning (3011087). Este curso abarca desde los
fundamentos teoricos de las redes neuronales totalmente conectadas hasta arquitecturas avanzadas
como CNNs, GANs, RNNs, GNNs y Transformers/LLMs.

## Objetivos del Curso

### Objetivo General

Introducir al participante en el conocimiento de las redes neuronales artificiales, brindando un
dominio de herramientas de ultima generacion para clasificacion, deteccion, segmentacion,
generacion de imagenes y produccion de texto.

### Objetivos Especificos

- Conocer las generalidades, componentes, arquitecturas y tipos de redes neuronales y sus
  aplicaciones reales.
- Aplicar Transfer Learning y tecnicas de anotacion de datos/imagenes para el entrenamiento de
  redes profundas (DNN).
- Desarrollar algoritmos de Deep Learning para tareas de clasificacion, deteccion de objetos,
  segmentacion y generacion de contenido.
- Utilizar entornos y librerias modernas de Python para visualizacion, procesamiento de datos y
  desarrollo de modelos de Inteligencia Artificial.

## Contenido del Programa

<!-- prettier-ignore -->
| Modulo | Tema /   Arquitectura | Descripcion    |
|---|---|---|
| 0. Introduccion   y FCN | Fully Connected Networks (FCN / MLP) | Fundamentos de redes neuronales, Teorema de   Aproximacion Universal, funciones de activacion, retropropagacion y   optimizacion. |
| 1. Vision por   Computador | Convolutional Neural Networks (CNN) | Extraccion de caracteristicas espaciales,   convoluciones, clasificacion, deteccion y segmentacion de imagenes. |
| 2. Modelos   Generativos | Generative Adversarial Networks (GAN) | Redes generativas antagonicas, arquitectura   generador-discriminador y sintesis de imagenes. |
| 3. Datos   Secuenciales | Recurrent Neural Networks (RNN) | Procesamiento de series temporales, secuencias   y modelos recurrentes (LSTM / GRU). |
| 4. Datos   Estructurados en Grafos | Graph Neural Networks (GNN) | Aprendizaje en estructuras no euclidianas y   relaciones complejas entre nodos. |
| 5. Modelos   Modernos de IA | Transformers, Atencion y LLMs | Mecanismos de atencion, arquitecturas   Transformer, modelos de lenguaje y procesamiento multimodal. |

## Esquema de Evaluacion

- **20% — FollowUps, Quizzes y Lecturas:** Seguimientos rapidos en clase y evaluaciones
  individuales de afianzamiento conceptual.
- **50% — WorkShops / Talleres Practicos:** Trabajos practicos asincronicos en equipo (grupos de
  maximo 5 integrantes).
- **30% — Proyecto Final:** Exposicion e implementacion grupal de una aplicacion real o tema de
  interes en Deep Learning.

## Tecnologias y Herramientas

- **Lenguaje:** Python 3.x
- **Frameworks principales:** PyTorch, TensorFlow, Keras, Fastai, Scikit-Learn.
- **Entornos de desarrollo:** Jupyter Notebook (local), Google Colab, VS Code, Kaggle.
- **Asistencia con IA:** Se promueve el uso guiado de asistentes de IA (ChatGPT, Claude,
  NotebookLM, etc.) para acelerar la experimentacion y el aprendizaje.

## Referencias y Bibliografia

1. Goodfellow, I., Bengio, Y., &amp; Courville, A. (2016). Deep Learning. MIT Press.
2. Cursos de Fastai (V1 y V2) — Jeremy Howard et al.
3. Deep Learning Specialization — Andrew Ng (Coursera).
4. Zeiler, M. D., &amp; Fergus, R. (2014). Visualizing and Understanding Convolutional Networks.
