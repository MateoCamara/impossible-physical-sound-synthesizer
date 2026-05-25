"""Rutas, hiperparametros y semillas centralizados."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
LABELS_CSV = DATA_DIR / "labels.csv"
SPLITS_JSON = DATA_DIR / "splits.json"

OUTPUTS_DIR = REPO_ROOT / "outputs"
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "figures"
PERCEPTUAL_DIR = REPO_ROOT / "perceptual_test"

SAMPLE_RATE = 44_100
CLIP_DURATION_S = 5.0
CLIP_SAMPLES = int(SAMPLE_RATE * CLIP_DURATION_S)

SEED = 20260520

# Encoder selection (resuelto en encoders/rave_wrapper.py)
PREFERRED_ENCODER = os.getenv("PREFERRED_ENCODER", "encodec")

RAVE_CHECKPOINT_PATH = os.getenv("RAVE_CHECKPOINT_PATH", "")
ENCODEC_CHECKPOINT_PATH = os.getenv("ENCODEC_CHECKPOINT_PATH", "")
CLAP_CHECKPOINT_PATH = os.getenv("CLAP_CHECKPOINT_PATH", "")

FREESOUND_API_KEY = os.getenv("FREESOUND_API_KEY", "")

# FAD fijado a VGGish (decision irrevocable del dia 1, ver plan)
FAD_EMBEDDING = "vggish"
