from __future__ import annotations

import os
import sys
import tempfile
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from gensim.models import Word2Vec, word2vec
from rdkit import Chem
from transformers import AutoModel, AutoTokenizer, BertModel, BertTokenizer
from unimol_tools import UniMolRepr

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]
EMBEDDING_BUNDLE_ROOT = PROJECT_ROOT / "apps" / "embedding_bundle"
SRC_ROOT = EMBEDDING_BUNDLE_ROOT / "src"
MODELS_DIR = Path(
    os.getenv(
        "EMBEDDING_MODELS_DIR",
        str(EMBEDDING_BUNDLE_ROOT / "models"),
    )
)
HF_CACHE_DIR = Path(
    os.getenv(
        "EMBEDDING_HF_CACHE_DIR",
        str(EMBEDDING_BUNDLE_ROOT / "hf_cache"),
    )
)
OUTPUT_ROOT = Path(
    os.getenv(
        "EMBEDDING_OUTPUT_DIR",
        str(PROJECT_ROOT / "apps" / "embedding_backend" / "outputs"),
    )
)

MOL2VEC_MODEL_PATH = MODELS_DIR / "model_300dim.pkl"
GIN_CHECKPOINT_PATH = MODELS_DIR / "grover_large.pt"
PROTVEC_MODEL_PATH = MODELS_DIR / "provec.model"

MOL2VEC_RADIUS = 1
GIN_FINGERPRINT_SOURCE = os.getenv("EMBEDDING_GIN_FINGERPRINT_SOURCE", "both")
ESM_MODEL_NAME = os.getenv("EMBEDDING_ESM_MODEL_NAME", "facebook/esm2_t33_650M_UR50D")
ESM_MAX_LENGTH = int(os.getenv("EMBEDDING_ESM_MAX_LENGTH", "1024"))
PROTBERT_MODEL_NAME = os.getenv("EMBEDDING_PROTBERT_MODEL_NAME", "Rostlab/prot_bert")
PROTBERT_MAX_LENGTH = int(os.getenv("EMBEDDING_PROTBERT_MAX_LENGTH", "1024"))
PROTVEC_KMER = 3
PROTVEC_VECTOR_SIZE = 100
DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")

MOL2VEC_DIR = SRC_ROOT / "Drug_Embeddings" / "Mol2Vec"
GIN_DIR = SRC_ROOT / "Drug_Embeddings" / "GIN"
for _path in [str(MOL2VEC_DIR), str(GIN_DIR)]:
    if _path not in sys.path:
        sys.path.append(_path)

os.environ["HF_HOME"] = str(HF_CACHE_DIR)

from features import MolSentence, mol2alt_sentence, sentences2vec  # noqa: E402
from task.fingerprint import generate_fingerprints  # noqa: E402


@dataclass
class EmbeddingResult:
    request_id: str
    metadata: Dict[str, str]
    vectors: Dict[str, List[float]]
    dimensions: Dict[str, int]
    artifacts: Dict[str, str]


class ModelCache:
    def __init__(self) -> None:
        self.unimol: Optional[UniMolRepr] = None
        self.mol2vec: Optional[Word2Vec] = None
        self.protvec: Optional[Word2Vec] = None
        self.esm_tokenizer: Any = None
        self.esm_model: Any = None
        self.protbert_tokenizer: Any = None
        self.protbert_model: Any = None


_CACHE = ModelCache()


def ensure_required_assets() -> None:
    required = [MOL2VEC_MODEL_PATH, GIN_CHECKPOINT_PATH, PROTVEC_MODEL_PATH]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required local model files:\n"
            + "\n".join(missing)
            + "\nRun: python3 apps/embedding_bundle/setup_bundle.py"
        )


def _load_unimol() -> UniMolRepr:
    if _CACHE.unimol is None:
        _CACHE.unimol = UniMolRepr(data_type="molecule", remove_hs=False, device=DEVICE)
    return _CACHE.unimol


def _load_mol2vec() -> Word2Vec:
    if _CACHE.mol2vec is None:
        _CACHE.mol2vec = word2vec.Word2Vec.load(str(MOL2VEC_MODEL_PATH))
    return _CACHE.mol2vec


def _load_protvec() -> Word2Vec:
    if _CACHE.protvec is None:
        _CACHE.protvec = Word2Vec.load(str(PROTVEC_MODEL_PATH))
    return _CACHE.protvec


def _load_esm() -> tuple[Any, Any]:
    if _CACHE.esm_tokenizer is None or _CACHE.esm_model is None:
        _CACHE.esm_tokenizer = AutoTokenizer.from_pretrained(
            ESM_MODEL_NAME, cache_dir=str(HF_CACHE_DIR), local_files_only=True
        )
        _CACHE.esm_model = AutoModel.from_pretrained(
            ESM_MODEL_NAME, cache_dir=str(HF_CACHE_DIR), local_files_only=True
        ).to(DEVICE)
        _CACHE.esm_model.eval()
    return _CACHE.esm_tokenizer, _CACHE.esm_model


def _load_protbert() -> tuple[Any, Any]:
    if _CACHE.protbert_tokenizer is None or _CACHE.protbert_model is None:
        _CACHE.protbert_tokenizer = BertTokenizer.from_pretrained(
            PROTBERT_MODEL_NAME,
            do_lower_case=False,
            cache_dir=str(HF_CACHE_DIR),
            local_files_only=True,
        )
        _CACHE.protbert_model = BertModel.from_pretrained(
            PROTBERT_MODEL_NAME, cache_dir=str(HF_CACHE_DIR), local_files_only=True
        ).to(DEVICE)
        _CACHE.protbert_model.eval()
    return _CACHE.protbert_tokenizer, _CACHE.protbert_model


def validate_inputs(canonical_smiles: str, gene_sequence: str) -> str:
    if not canonical_smiles or Chem.MolFromSmiles(canonical_smiles) is None:
        raise ValueError(f"Invalid canonical_smiles: {canonical_smiles}")
    clean_sequence = gene_sequence.replace(" ", "").strip()
    if not clean_sequence:
        raise ValueError("gene_sequence must be a non-empty amino-acid sequence.")
    return clean_sequence


def _drug_unimol(smiles: str) -> np.ndarray:
    model = _load_unimol()
    output = model.get_repr([smiles])
    if isinstance(output, dict):
        return output["cls_repr"].cpu().numpy()[0]
    if isinstance(output, list):
        return np.asarray(output[0])
    raise ValueError(f"Unexpected Uni-Mol output type: {type(output)}")


def _drug_mol2vec(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid canonical_smiles for Mol2Vec: {smiles}")
    model = _load_mol2vec()
    sentence = MolSentence(mol2alt_sentence(mol, MOL2VEC_RADIUS))
    return np.asarray(sentences2vec([sentence], model, unseen="UNK")[0])


def _build_gin_args(temp_csv_path: str, temp_output_path: str):
    from argparse import Namespace

    args = Namespace()
    args.data_path = temp_csv_path
    args.output_path = temp_output_path
    args.checkpoint_path = str(GIN_CHECKPOINT_PATH)
    args.checkpoint_paths = [str(GIN_CHECKPOINT_PATH)]
    args.features_path = None
    args.fingerprint_source = GIN_FINGERPRINT_SOURCE
    args.no_cuda = DEVICE != "cuda"
    args.cuda = DEVICE == "cuda"
    args.batch_size = 32
    args.max_data_size = float("inf")
    args.use_compound_names = False
    args.no_cache = True
    args.bond_drop_rate = 0
    args.parser_name = "fingerprint"
    return args


def _drug_gin(smiles: str) -> np.ndarray:
    if GIN_FINGERPRINT_SOURCE not in {"atom", "bond", "both"}:
        raise ValueError('EMBEDDING_GIN_FINGERPRINT_SOURCE must be "atom", "bond", or "both".')
    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_csv = Path(tmp_dir) / "single_smiles.csv"
        temp_out = Path(tmp_dir) / "tmp.csv"
        pd.DataFrame({"smiles": [smiles]}).to_csv(temp_csv, index=False)
        args = _build_gin_args(str(temp_csv), str(temp_out))
        fingerprints = generate_fingerprints(args)
    if not fingerprints:
        raise RuntimeError("GIN/GROVER did not generate a fingerprint.")
    return np.asarray(fingerprints[0])


def _gene_esm(sequence: str) -> np.ndarray:
    tokenizer, model = _load_esm()
    inputs = tokenizer(
        [sequence],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=ESM_MAX_LENGTH,
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 1:-1, :].mean(dim=1).cpu().numpy()[0]


def _gene_protbert(sequence: str) -> np.ndarray:
    tokenizer, model = _load_protbert()
    spaced_sequence = " ".join(list(sequence))
    inputs = tokenizer(
        [spaced_sequence],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=PROTBERT_MAX_LENGTH,
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 1:-1, :].mean(dim=1).cpu().numpy()[0]


def _gene_protvec(sequence: str) -> np.ndarray:
    if len(sequence) < PROTVEC_KMER:
        raise ValueError(f"gene_sequence length must be >= {PROTVEC_KMER} for ProtVec.")
    model = _load_protvec()
    kmers = [sequence[i : i + PROTVEC_KMER] for i in range(len(sequence) - PROTVEC_KMER + 1)]
    vectors = [model.wv[kmer] for kmer in kmers if kmer in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    return np.zeros(PROTVEC_VECTOR_SIZE, dtype=float)


def _to_float_list(v: np.ndarray) -> List[float]:
    return [float(x) for x in v.tolist()]


def _build_rows(metadata: Dict[str, str], vectors: Dict[str, List[float]]) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    drug_row: Dict[str, Any] = {
        "drug_id": metadata["drug_id"],
        "canonical_smiles": metadata["canonical_smiles"],
    }
    drug_row.update({f"drug_unimol_{i}": v for i, v in enumerate(vectors["drug_unimol"])})
    drug_row.update({f"drug_mol2vec_{i}": v for i, v in enumerate(vectors["drug_mol2vec"])})
    drug_row.update({f"drug_gin_{i}": v for i, v in enumerate(vectors["drug_gin"])})

    gene_row: Dict[str, Any] = {
        "gene_id": metadata["gene_id"],
        "gene_sequence": metadata["gene_sequence"],
    }
    gene_row.update({f"gene_esm_{i}": v for i, v in enumerate(vectors["gene_esm"])})
    gene_row.update({f"gene_protbert_{i}": v for i, v in enumerate(vectors["gene_protbert"])})
    gene_row.update({f"gene_protvec_{i}": v for i, v in enumerate(vectors["gene_protvec"])})

    metadata_row = dict(metadata)
    combined_row: Dict[str, Any] = {}
    combined_row.update(metadata_row)
    combined_row.update({k: v for k, v in drug_row.items() if k not in metadata_row})
    combined_row.update({k: v for k, v in gene_row.items() if k not in metadata_row})
    return metadata_row, drug_row, gene_row, combined_row


def _write_artifacts(
    user_id: str,
    request_id: str,
    metadata_row: Dict[str, Any],
    drug_row: Dict[str, Any],
    gene_row: Dict[str, Any],
    combined_row: Dict[str, Any],
    include_combined_csv: bool,
    create_zip: bool,
) -> Dict[str, str]:
    output_dir = OUTPUT_ROOT / str(user_id) / request_id
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_csv = output_dir / "input_metadata.csv"
    drug_csv = output_dir / "drug_embeddings.csv"
    gene_csv = output_dir / "gene_embeddings.csv"
    combined_csv = output_dir / "combined_embeddings.csv"
    zip_path = output_dir / "embeddings_bundle.zip"

    pd.DataFrame([metadata_row]).to_csv(metadata_csv, index=False)
    pd.DataFrame([drug_row]).to_csv(drug_csv, index=False)
    pd.DataFrame([gene_row]).to_csv(gene_csv, index=False)
    if include_combined_csv:
        pd.DataFrame([combined_row]).to_csv(combined_csv, index=False)

    artifacts = {
        "output_dir": str(output_dir),
        "input_metadata_csv": str(metadata_csv),
        "drug_embeddings_csv": str(drug_csv),
        "gene_embeddings_csv": str(gene_csv),
    }
    if include_combined_csv:
        artifacts["combined_embeddings_csv"] = str(combined_csv)

    if create_zip:
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(metadata_csv, arcname=metadata_csv.name)
            zf.write(drug_csv, arcname=drug_csv.name)
            zf.write(gene_csv, arcname=gene_csv.name)
            if include_combined_csv:
                zf.write(combined_csv, arcname=combined_csv.name)
        artifacts["zip_file"] = str(zip_path)

    return artifacts


def compute_embeddings(
    *,
    drug_id: str,
    canonical_smiles: str,
    gene_id: str,
    gene_sequence: str,
    user_id: str,
    request_id: Optional[str] = None,
    include_vectors: bool = True,
    include_combined_csv: bool = True,
    create_zip: bool = True,
) -> EmbeddingResult:
    ensure_required_assets()
    clean_sequence = validate_inputs(canonical_smiles, gene_sequence)
    rid = request_id or str(uuid.uuid4())

    drug_unimol = _drug_unimol(canonical_smiles)
    drug_mol2vec = _drug_mol2vec(canonical_smiles)
    drug_gin = _drug_gin(canonical_smiles)
    gene_esm = _gene_esm(clean_sequence)
    gene_protbert = _gene_protbert(clean_sequence)
    gene_protvec = _gene_protvec(clean_sequence)

    metadata = {
        "drug_id": drug_id.strip(),
        "canonical_smiles": canonical_smiles.strip(),
        "gene_id": gene_id.strip(),
        "gene_sequence": clean_sequence,
    }
    vectors = {
        "drug_unimol": _to_float_list(drug_unimol),
        "drug_mol2vec": _to_float_list(drug_mol2vec),
        "drug_gin": _to_float_list(drug_gin),
        "gene_esm": _to_float_list(gene_esm),
        "gene_protbert": _to_float_list(gene_protbert),
        "gene_protvec": _to_float_list(gene_protvec),
    }
    dimensions = {k: len(v) for k, v in vectors.items()}

    metadata_row, drug_row, gene_row, combined_row = _build_rows(metadata, vectors)
    artifacts = _write_artifacts(
        user_id=user_id,
        request_id=rid,
        metadata_row=metadata_row,
        drug_row=drug_row,
        gene_row=gene_row,
        combined_row=combined_row,
        include_combined_csv=include_combined_csv,
        create_zip=create_zip,
    )

    return EmbeddingResult(
        request_id=rid,
        metadata=metadata,
        vectors=vectors if include_vectors else {},
        dimensions=dimensions,
        artifacts=artifacts,
    )

