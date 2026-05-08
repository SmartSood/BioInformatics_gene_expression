from __future__ import annotations

from typing import Any, Dict

from apps.embedding_backend.services.embedding_service import compute_embeddings


def run_embedding_job(payload: Dict[str, Any], user_id: str) -> Dict[str, Any]:
    result = compute_embeddings(
        drug_id=payload["drug_id"],
        canonical_smiles=payload["canonical_smiles"],
        gene_id=payload["gene_id"],
        gene_sequence=payload["gene_sequence"],
        user_id=user_id,
        request_id=payload.get("request_id"),
        include_vectors=bool(payload.get("include_vectors", False)),
        include_combined_csv=bool(payload.get("include_combined_csv", True)),
        create_zip=bool(payload.get("create_zip", True)),
    )
    return {
        "request_id": result.request_id,
        "metadata": result.metadata,
        "dimensions": result.dimensions,
        "artifacts": result.artifacts,
        "vectors": result.vectors if payload.get("include_vectors", False) else {},
    }

