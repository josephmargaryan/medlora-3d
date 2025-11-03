"""Project-wide constants.

Keep this file minimal: simple, stable values used across the package. Prefer
reading an environment variable for user-specific paths (see MEDLORA_SSL_PATH).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple, Sequence


# full list (kept as list because it's used for argparse choices / ordering)
MSD_TASKS: Sequence[str] = [
    "Task01_BrainTumour",
    "Task02_Heart",
    "Task03_Liver",
    "Task04_Hippocampus",
    "Task05_Prostate",
    "Task06_Lung",
    "Task07_Pancreas",
    "Task08_HepaticVessel",
    "Task09_Spleen",
    "Task10_Colon",
]

# sets used only for membership tests; make them immutable to avoid accidental mutation
CT_TASKS = frozenset(
    [
        "Task03_Liver",
        "Task06_Lung",
        "Task07_Pancreas",
        "Task08_HepaticVessel",
        "Task09_Spleen",
        "Task10_Colon",
    ]
)
MRI_TASKS = frozenset(
    [
        "Task01_BrainTumour",
        "Task02_Heart",
        "Task04_Hippocampus",
        "Task05_Prostate",
    ]
)

DEFAULT_VAL_FRAC = 0.2
DEFAULT_ROI: Tuple[int, int, int] = (96, 96, 96)

# SSL pre-trained weights (used by models.ensure_ssl / load_ct_ssl_encoder)
SSL_URL = (
    "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/"
    "ssl_pretrained_weights.pth"
)

# Default path for the downloaded SSL weights. 
# We should allow users to override this via the MEDLORA_SSL_PATH environment variable.
# But for colab usage this might suffice?
SSL_PATH = os.environ.get(
    "MEDLORA_SSL_PATH", str(Path.home() / ".medlora" / "ssl_pretrained_weights.pth")
)
