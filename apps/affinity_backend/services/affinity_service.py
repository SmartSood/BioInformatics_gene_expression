from __future__ import annotations

import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "apps" / "affinity" / "gene_embeddings.pth"
CHECKPOINT_PATH = Path(os.getenv("AFFINITY_CHECKPOINT_PATH", str(DEFAULT_CHECKPOINT)))
OUTPUT_ROOT = Path(
    os.getenv(
        "AFFINITY_OUTPUT_DIR",
        str(PROJECT_ROOT / "apps" / "affinity_backend" / "outputs"),
    )
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AffinityModel(nn.Module):
    def __init__(self, drug_dim: int, prot_dim: int):
        super().__init__()
        self.drug_proj = nn.Sequential(
            nn.Linear(drug_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
        )
        self.prot_proj = nn.Sequential(
            nn.Linear(prot_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
        )

        self.cnn = nn.Sequential(
            nn.Conv1d(1, 64, 5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, 3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
        )

        self.regressor = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, drug: torch.Tensor, protein: torch.Tensor) -> torch.Tensor:
        drug = F.normalize(drug, p=2, dim=1)
        protein = F.normalize(protein, p=2, dim=1)
        drug = self.drug_proj(drug)
        protein = self.prot_proj(protein)
        x = torch.cat([drug, protein], dim=1).unsqueeze(1)
        x = self.cnn(x).squeeze(-1)
        out = self.regressor(x)
        return out.squeeze()


@dataclass
class ModelBundle:
    model: Optional[AffinityModel]
    drug_dim: int
    prot_dim: int
    loaded: bool
    reason: Optional[str]


_BUNDLE: Optional[ModelBundle] = None


def _extract_dims_from_checkpoint(checkpoint: Dict) -> Tuple[int, int]:
    drug_dim = checkpoint.get("drug_dim")
    prot_dim = checkpoint.get("prot_dim")
    if isinstance(drug_dim, int) and isinstance(prot_dim, int):
        return drug_dim, prot_dim
    raise ValueError("checkpoint missing integer fields 'drug_dim' and 'prot_dim'")


def _load_bundle() -> ModelBundle:
    global _BUNDLE
    if _BUNDLE is not None:
        return _BUNDLE

    if not CHECKPOINT_PATH.exists():
        _BUNDLE = ModelBundle(
            model=None,
            drug_dim=0,
            prot_dim=0,
            loaded=False,
            reason=f"Checkpoint not found: {CHECKPOINT_PATH}",
        )
        return _BUNDLE

    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        if not isinstance(checkpoint, dict):
            raise ValueError("checkpoint is not a dict")

        drug_dim, prot_dim = _extract_dims_from_checkpoint(checkpoint)
        model_state = checkpoint.get("model_state_dict")
        if not isinstance(model_state, dict):
            raise ValueError("checkpoint missing 'model_state_dict'")

        model = AffinityModel(drug_dim=drug_dim, prot_dim=prot_dim).to(DEVICE)
        model.load_state_dict(model_state)
        model.eval()

        _BUNDLE = ModelBundle(
            model=model,
            drug_dim=drug_dim,
            prot_dim=prot_dim,
            loaded=True,
            reason=None,
        )
    except Exception as exc:
        _BUNDLE = ModelBundle(
            model=None,
            drug_dim=0,
            prot_dim=0,
            loaded=False,
            reason=f"Failed to load checkpoint: {exc}",
        )
    return _BUNDLE


def _sorted_feature_columns(cols: Iterable[str], prefix: str) -> List[str]:
    p = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    indexed: List[Tuple[int, str]] = []
    for col in cols:
        m = p.match(col)
        if m:
            indexed.append((int(m.group(1)), col))
    indexed.sort(key=lambda x: x[0])
    return [c for _, c in indexed]


def required_feature_prefixes() -> List[str]:
    return [
        "drug_mol2vec_",
        "drug_gin_",
        "drug_unimol_",
        "gene_protvec_",
        "gene_protbert_",
        "gene_esm_",
    ]


def _resolve_group_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    cols = list(df.columns)
    mapping = {
        "drug_mol2vec": _sorted_feature_columns(cols, "drug_mol2vec_"),
        "drug_gin": _sorted_feature_columns(cols, "drug_gin_"),
        "drug_unimol": _sorted_feature_columns(cols, "drug_unimol_"),
        "gene_protvec": _sorted_feature_columns(cols, "gene_protvec_"),
        "gene_protbert": _sorted_feature_columns(cols, "gene_protbert_"),
        "gene_esm": _sorted_feature_columns(cols, "gene_esm_"),
    }
    return mapping


def _validate_input_df(df: pd.DataFrame, bundle: ModelBundle) -> Dict[str, List[str]]:
    if "drug_id" not in df.columns:
        raise ValueError("CSV must include 'drug_id' column")
    if "gene_id" not in df.columns and "protein_id" not in df.columns:
        raise ValueError("CSV must include either 'gene_id' or 'protein_id' column")

    groups = _resolve_group_columns(df)
    missing_groups = [k for k, v in groups.items() if len(v) == 0]
    if missing_groups:
        raise ValueError(
            "Missing embedding columns for: " + ", ".join(missing_groups)
        )

    return groups


def _align_feature_dim(matrix: np.ndarray, expected_dim: int) -> np.ndarray:
    """Align feature width to the model checkpoint by truncating or zero-padding.

    This keeps prediction usable when CSV vectors are wider/narrower than the
    training checkpoint dimensions.
    """
    current_dim = matrix.shape[1]
    if current_dim == expected_dim:
        return matrix
    if current_dim > expected_dim:
        return matrix[:, :expected_dim]

    pad = np.zeros((matrix.shape[0], expected_dim - current_dim), dtype=matrix.dtype)
    return np.concatenate([matrix, pad], axis=1)


def predict_affinity(df: pd.DataFrame) -> pd.DataFrame:
    bundle = _load_bundle()
    if not bundle.loaded or bundle.model is None:
        raise RuntimeError(bundle.reason or "Affinity model is not loaded")

    groups = _validate_input_df(df, bundle)

    drug_matrix = np.concatenate(
        [
            df[groups["drug_mol2vec"]].to_numpy(dtype=np.float32),
            df[groups["drug_gin"]].to_numpy(dtype=np.float32),
            df[groups["drug_unimol"]].to_numpy(dtype=np.float32),
        ],
        axis=1,
    )
    prot_matrix = np.concatenate(
        [
            df[groups["gene_protvec"]].to_numpy(dtype=np.float32),
            df[groups["gene_protbert"]].to_numpy(dtype=np.float32),
            df[groups["gene_esm"]].to_numpy(dtype=np.float32),
        ],
        axis=1,
    )

    # Auto-align to checkpoint dims so inference remains robust to embedding
    # variants (for example, wider GIN feature exports).
    drug_matrix = _align_feature_dim(drug_matrix, bundle.drug_dim)
    prot_matrix = _align_feature_dim(prot_matrix, bundle.prot_dim)

    with torch.no_grad():
        drug_t = torch.from_numpy(drug_matrix).to(DEVICE)
        prot_t = torch.from_numpy(prot_matrix).to(DEVICE)
        preds = bundle.model(drug_t, prot_t).detach().cpu().numpy()

    out = df.copy()
    out["predicted_affinity"] = preds.astype(float)
    return out


def save_prediction_csv(user_id: str, drug_name: str, gene_name: str, df_out: pd.DataFrame) -> Tuple[str, Path]:
    request_id = str(uuid.uuid4())
    user_dir = OUTPUT_ROOT / user_id / request_id
    user_dir.mkdir(parents=True, exist_ok=True)

    def sanitize(s: str) -> str:
        return re.sub(r"[^A-Za-z0-9_\-.]", "", re.sub(r"\s+", "_", (s or "").strip()))

    safe_drug = sanitize(drug_name or "drug")
    safe_gene = sanitize(gene_name or "gene")
    filename = f"{safe_drug}_{safe_gene}_affinity_predictions.csv"
    out_path = user_dir / filename
    df_out.to_csv(out_path, index=False)
    return request_id, out_path


def build_sample_csv() -> pd.DataFrame:
    bundle = _load_bundle()
    if not bundle.loaded:
        # fallback minimum sample with expected prefixes when checkpoint is unavailable
        data = {
            "drug_id": ["EXAMPLE_DRUG"],
            "gene_id": ["EXAMPLE_GENE"],
            "drug_mol2vec_0": [0.0],
            "drug_gin_0": [0.0],
            "drug_unimol_0": [0.0],
            "gene_protvec_0": [0.0],
            "gene_protbert_0": [0.0],
            "gene_esm_0": [0.0],
        }
        return pd.DataFrame(data)

    # if model is loaded, create a valid-shape zero vector sample
    # split dims based on common embedding dim conventions from the notebook order.
    # this is only for template guidance.
    d1 = min(300, bundle.drug_dim)
    d2 = min(300, max(bundle.drug_dim - d1, 0))
    d3 = max(bundle.drug_dim - d1 - d2, 0)
    p1 = min(100, bundle.prot_dim)
    p2 = min(1024, max(bundle.prot_dim - p1, 0))
    p3 = max(bundle.prot_dim - p1 - p2, 0)

    row: Dict[str, float | str] = {
        "drug_id": "EXAMPLE_DRUG",
        "gene_id": "EXAMPLE_GENE",
    }
    for i in range(d1):
        row[f"drug_mol2vec_{i}"] = 0.0
    for i in range(d2):
        row[f"drug_gin_{i}"] = 0.0
    for i in range(d3):
        row[f"drug_unimol_{i}"] = 0.0
    for i in range(p1):
        row[f"gene_protvec_{i}"] = 0.0
    for i in range(p2):
        row[f"gene_protbert_{i}"] = 0.0
    for i in range(p3):
        row[f"gene_esm_{i}"] = 0.0
    return pd.DataFrame([row])
