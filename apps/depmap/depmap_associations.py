import argparse
import csv
import os
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Union

import numpy as np
import pandas as pd


###############################################################################
# Paths and filenames expected in your local depmap clone
#
# You can override any of these via command-line flags, but this shows the
# default layout the script expects (all files placed in the same directory
# as this script).
##############################################################################################################################################################
# DATASET FILE PATHS - UPDATE THESE WHEN NEW VERSIONS BECOME AVAILABLE
###############################################################################

# Base directory for all data files. Anchored to this script's own location
# (not the process's cwd, which depmap_worker.py has no reason to set to
# apps/depmap/ - it imports this module via sys.path insertion and runs
# from /app). A bare relative "dataset" string here silently resolved to
# nothing under any other cwd, so every gene lookup found an empty
# DataFrame instead of erroring - it looked like "gene not found" rather
# than "file not found".
DATASET_DIR = str(Path(__file__).resolve().parent / "dataset")

# ============================================================================
# EXPRESSION DATA
# ============================================================================
# Location: dataset/expression/
# Current version: 25Q3 (as of 2025-09-25)
# To update: Download new version from downloads.csv and replace this file
# Latest URL: Check downloads.csv for "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv" with latest release
DEFAULT_EXPRESSION_FILE = os.path.join(DATASET_DIR, "expression", "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv")

# ============================================================================
# MODEL METADATA (Cell line mapping)
# ============================================================================
# Location: dataset/model/
# To update: Download new Model.csv from DepMap portal
DEFAULT_MODEL_FILE = os.path.join(DATASET_DIR, "model", "Model.csv")

# ============================================================================
# GDSC DRUG SENSITIVITY DATA
# ============================================================================
# Location: dataset/gdsc/
# To update: Download new sanger-dose-response.csv from DepMap portal
DEFAULT_GDSC_DOSE_RESPONSE_FILE = os.path.join(DATASET_DIR, "gdsc", "sanger-dose-response.csv")

# ============================================================================
# CTRP DRUG SENSITIVITY DATA
# ============================================================================
# Location: dataset/ctrp/
# To update: Download new CTRP zip file and replace this file
DEFAULT_CTRP_ZIP_FILE = os.path.join(DATASET_DIR, "ctrp", "CTRPv2.0_2015_ctd2_ExpandedDataset.zip")

# ============================================================================
# PRISM REPURPOSING SECONDARY SCREEN
# ============================================================================
# Location: dataset/prism_secondary/
# To update: Download new secondary screen files from DepMap portal
DEFAULT_PRISM_SECONDARY_MATRIX_FILE = os.path.join(DATASET_DIR, "prism_secondary", "secondary-screen-replicate-collapsed-logfold-change.csv")
DEFAULT_PRISM_SECONDARY_TREATMENT_INFO_FILE = os.path.join(DATASET_DIR, "prism_secondary", "secondary-screen-replicate-collapsed-treatment-info.csv")
DEFAULT_PRISM_SECONDARY_CELL_INFO_FILE = os.path.join(DATASET_DIR, "prism_secondary", "secondary-screen-cell-line-info.csv")

# ============================================================================
# PRISM REPURPOSING PUBLIC DATASET
# ============================================================================
# Location: dataset/prism_public/
# Current version: 24Q2 (as of 2024-05-28)
# To update: 
#   1. Download new PRISM Public files (e.g., 25Q3 when available) from downloads.csv
#   2. Update filenames below to match new version (e.g., Repurposing_Public_25Q3_*.csv)
#   3. Update dataset_label in _compute_correlations call (line ~872) to match new version
# Latest URL: Check downloads.csv for "Repurposing_Public_*" with latest release
DEFAULT_PRISM_PUBLIC_MATRIX_FILE = os.path.join(DATASET_DIR, "prism_public", "Repurposing_Public_24Q2_LFC_COLLAPSED.csv")
DEFAULT_PRISM_PUBLIC_TREATMENT_INFO_FILE = os.path.join(DATASET_DIR, "prism_public", "Repurposing_Public_24Q2_Treatment_Meta_Data.csv")
DEFAULT_PRISM_PUBLIC_CELL_INFO_FILE = os.path.join(DATASET_DIR, "prism_public", "Repurposing_Public_24Q2_Cell_Line_Meta_Data.csv")


###############################################################################
# Utility helpers
###############################################################################


def _load_expression(
    expression_path: str, genes: List[str]
) -> pd.DataFrame:
    """
    Load expression for the requested genes.

    The OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv file has one row per
    cell line (indexed by ModelID, which is the same as DepMap_ID) and one column per gene.
    Gene columns are in format "GENE (ID)" (e.g., "EGFR (1956)").
    """
    # The expression file uses ModelID as the identifier (e.g., ACH-000001)
    # ModelID is the same as DepMap_ID in DepMap terminology
    # Gene columns are in format "GENE (ID)", so we need to match flexibly
    
    # First, read just the header to find matching gene columns
    header_df = pd.read_csv(expression_path, nrows=0)
    all_cols = header_df.columns.tolist()
    
    # Find ModelID column
    modelid_col = None
    for col in all_cols:
        if col == "ModelID":
            modelid_col = col
            break
    
    if modelid_col is None:
        raise ValueError("Expression file missing 'ModelID' column")
    
    # Find gene columns - match either exact name or "GENE (ID)" format
    gene_cols = []
    for gene in genes:
        # Try exact match first
        if gene in all_cols:
            gene_cols.append(gene)
        else:
            # Try matching "GENE (ID)" format
            found = False
            for col in all_cols:
                # Match if column starts with "GENE (" or equals "GENE"
                if col.startswith(f"{gene} (") or col == gene:
                    gene_cols.append(col)
                    found = True
                    break
            if not found:
                print(f"Warning: Gene '{gene}' not found in expression file. Available columns contain: {[c for c in all_cols if gene.lower() in c.lower()][:5]}")
    
    if not gene_cols:
        raise ValueError(f"None of the requested genes {genes} found in expression file")
    
    usecols = [modelid_col] + gene_cols
    
    df = pd.read_csv(
        expression_path,
        usecols=usecols,
        low_memory=False,
    )

    df = df.dropna(subset=[modelid_col])
    # Rename ModelID to DepMap_ID for consistency with rest of code
    df = df.rename(columns={modelid_col: "DepMap_ID"})
    
    # Rename gene columns to just the gene name (remove " (ID)" suffix)
    rename_dict = {}
    for col in gene_cols:
        if " (" in col:
            # Extract gene name before parentheses
            gene_name = col.split(" (")[0]
            rename_dict[col] = gene_name
        else:
            rename_dict[col] = col
    
    df = df.rename(columns=rename_dict)
    df = df.set_index("DepMap_ID")
    
    # Handle duplicate indices: if there are multiple rows per DepMap_ID,
    # take the mean (or first non-null value) for each gene
    if df.index.duplicated().any():
        # Group by DepMap_ID and take mean of expression values
        # This handles cases where the same cell line has multiple sequencing runs
        df = df.groupby(df.index).mean()
    
    return df


def _load_model_metadata(model_path: str) -> pd.DataFrame:
    """
    Load Model.csv which contains mappings between DepMap_ID, CCLE_Name,
    COSMIC_ID, etc.
    """
    model = pd.read_csv(model_path, low_memory=False)
    return model


###############################################################################
# GDSC (Sanger) drug sensitivity
###############################################################################


def _load_gdsc_auc(
    gdsc_path: str,
    model: pd.DataFrame,
) -> List[Tuple[pd.DataFrame, str]]:
    """
    Load GDSC dose-response data and return separate matrices for GDSC1 and GDSC2.
    Returns a list of (dataframe, dataset_label) tuples.

    Expected columns in sanger-dose-response.csv:
      - 'DATASET' (GDSC1 or GDSC2)
      - 'DRUG_NAME' or 'drug_name'
      - 'DRUG_ID' or 'drug_id'
      - 'AUC' or 'auc'
      - 'COSMIC_ID' or 'cosmic_id'
    """
    df = pd.read_csv(gdsc_path, low_memory=False)

    # Normalize column names
    df.columns = [c.lower() for c in df.columns]

    # Check for required columns
    if "dataset" not in df.columns:
        raise ValueError("sanger-dose-response.csv must contain 'DATASET' column")
    
    if "cosmic_id" not in df.columns:
        raise ValueError("sanger-dose-response.csv must contain 'COSMIC_ID' column")

    # Map COSMIC_ID -> DepMap_ID (ModelID) via Model.csv
    model_tmp = model.copy()
    model_tmp.columns = [c.lower() for c in model_tmp.columns]
    if "cosmicid" not in model_tmp.columns or "modelid" not in model_tmp.columns:
        raise ValueError(
            "Model.csv does not contain expected columns 'COSMICID' and 'ModelID'"
        )
    cosmic_to_modelid = (
        model_tmp[["cosmicid", "modelid"]]
        .dropna()
        .drop_duplicates()
        .set_index("cosmicid")["modelid"]
    )
    df = df[df["cosmic_id"].isin(cosmic_to_modelid.index)].copy()
    df["depmap_id"] = df["cosmic_id"].map(cosmic_to_modelid)

    # Drug name and ID
    if "drug_name" not in df.columns:
        raise ValueError("sanger-dose-response.csv must contain 'DRUG_NAME' column")
    
    if "drug_id" not in df.columns:
        raise ValueError("sanger-dose-response.csv must contain 'DRUG_ID' column")

    # Use AUC column (prefer 'auc' over 'auc_published')
    if "auc" in df.columns:
        auc_col = "auc"
    elif "auc_published" in df.columns:
        auc_col = "auc_published"
    else:
        raise ValueError("sanger-dose-response.csv must contain an 'AUC' column")

    results = []
    
    # Process each dataset separately (GDSC1 and GDSC2)
    for dataset_name in ["GDSC1", "GDSC2"]:
        df_subset = df[df["dataset"] == dataset_name].copy()
        
        if len(df_subset) == 0:
            continue  # Skip if no data for this dataset
        
        # Create compound label with ID: "DRUG_NAME (GDSC1:DRUG_ID)"
        df_subset["compound_label"] = (
            df_subset["drug_name"].astype(str) 
            + " (" + dataset_name + ":" + df_subset["drug_id"].astype(str) + ")"
        )
        
        # Pivot to matrix: rows=DepMap_ID, columns=Compound, values=AUC
        pivot = (
            df_subset[["depmap_id", "compound_label", auc_col]]
            .dropna(subset=["depmap_id", "compound_label"])
            .drop_duplicates(subset=["depmap_id", "compound_label"])
            .pivot(index="depmap_id", columns="compound_label", values=auc_col)
        )
        
        pivot.index.name = "DepMap_ID"
        
        # Dataset label
        dataset_label = f"Drug sensitivity AUC (Sanger {dataset_name})"
        results.append((pivot, dataset_label))
    
    return results


###############################################################################
# CTRP drug sensitivity
###############################################################################


def _open_ctrp_curves_from_zip(zip_path: str) -> pd.DataFrame:
    """
    CTRPv2.0_2015_ctd2_ExpandedDataset.zip contains multiple text files.
    We want v20.data.curves_post_qc.txt (dose-response curves) which has
    an 'area_under_curve' field.
    """
    with zipfile.ZipFile(zip_path) as zf:
        target_name = None
        for name in zf.namelist():
            lower = name.lower()
            if "v20.data.curves_post_qc" in lower:
                target_name = name
                break
        if target_name is None:
            raise FileNotFoundError(
                "Could not find v20.data.curves_post_qc file inside CTRP zip"
            )

        with zf.open(target_name) as fh:
            # This file is tab-delimited
            df = pd.read_csv(fh, sep="\t", low_memory=False)
    return df


def _load_ctrp_auc(
    ctrp_zip_path: str,
    model: pd.DataFrame,
) -> pd.DataFrame:
    """
    Load CTRP curves data and pivot into DepMap_ID x Compound (AUC).

    We:
      - read v20.data.curves_post_qc (has experiment_id, master_cpd_id, area_under_curve)
      - join with v20.meta.per_experiment.txt to get master_ccl_id
      - join with v20.meta.per_cell_line.txt to get ccl_name (CCLE name)
      - join with v20.meta.per_compound.txt to get cpd_name
      - use 'area_under_curve' as the sensitivity metric
      - map 'ccl_name' to DepMap_ID via Model.csv
    """
    with zipfile.ZipFile(ctrp_zip_path) as zf:
        # Load curves data
        curves = _open_ctrp_curves_from_zip(ctrp_zip_path)
        curves.columns = [c.lower() for c in curves.columns]
        
        # Load metadata files
        exp_meta = pd.read_csv(zf.open("v20.meta.per_experiment.txt"), sep="\t", low_memory=False)
        exp_meta.columns = [c.lower() for c in exp_meta.columns]
        
        ccl_meta = pd.read_csv(zf.open("v20.meta.per_cell_line.txt"), sep="\t", low_memory=False)
        ccl_meta.columns = [c.lower() for c in ccl_meta.columns]
        
        cpd_meta = pd.read_csv(zf.open("v20.meta.per_compound.txt"), sep="\t", low_memory=False)
        cpd_meta.columns = [c.lower() for c in cpd_meta.columns]
    
    # Join curves with experiment metadata to get master_ccl_id
    curves = curves.merge(
        exp_meta[["experiment_id", "master_ccl_id"]],
        on="experiment_id",
        how="left"
    )
    
    # Join with cell line metadata to get ccl_name
    curves = curves.merge(
        ccl_meta[["master_ccl_id", "ccl_name"]],
        on="master_ccl_id",
        how="left"
    )
    
    # Join with compound metadata to get cpd_name
    curves = curves.merge(
        cpd_meta[["master_cpd_id", "cpd_name"]],
        on="master_cpd_id",
        how="left"
    )
    
    # Rename ccl_name to ccle_name for consistency
    curves = curves.rename(columns={"ccl_name": "ccle_name"})
    
    # Check required columns
    required_cols = {"master_cpd_id", "cpd_name", "ccle_name", "area_under_curve"}
    missing = required_cols - set(curves.columns)
    if missing:
        raise ValueError(
            f"CTRP data is missing required columns after joining metadata: {', '.join(sorted(missing))}"
        )

    model_tmp = model.copy()
    model_tmp.columns = [c.lower() for c in model_tmp.columns]
    # Model.csv uses "cclename" (no underscore) after lowercasing
    if "cclename" not in model_tmp.columns or "modelid" not in model_tmp.columns:
        raise ValueError(
            "Model.csv does not contain expected columns 'CCLEName' and 'ModelID'"
        )

    # CTRP uses short cell line names (e.g., "253J") while Model.csv uses full CCLE names (e.g., "253J_URINARY_TRACT")
    # Extract base name (part before first underscore) from CCLE names for matching
    model_tmp["cclename_base"] = model_tmp["cclename"].str.split("_").str[0]
    
    # Create mapping from base CCLE name to DepMap_ID (ModelID)
    # If multiple rows have the same base name, we'll need to handle duplicates
    ccle_base_to_depmap = (
        model_tmp[["cclename_base", "modelid"]]
        .dropna()
        .drop_duplicates(subset=["cclename_base", "modelid"])
    )
    
    # Handle cases where same base name maps to multiple ModelIDs (take first)
    ccle_base_to_depmap = ccle_base_to_depmap.groupby("cclename_base").first()["modelid"]

    # Match CTRP ccl_name (which is the short name) with base CCLE names
    curves = curves[curves["ccle_name"].isin(ccle_base_to_depmap.index)].copy()
    curves["DepMap_ID"] = curves["ccle_name"].map(ccle_base_to_depmap)

    # Construct a readable compound label similar to the portal
    curves["compound_label"] = (
        curves["cpd_name"].astype(str)
        + " (CTRP:"
        + curves["master_cpd_id"].astype(str)
        + ")"
    )

    pivot = (
        curves[["DepMap_ID", "compound_label", "area_under_curve"]]
        .dropna(subset=["DepMap_ID", "compound_label"])
        .drop_duplicates(subset=["DepMap_ID", "compound_label"])
        .pivot(index="DepMap_ID", columns="compound_label", values="area_under_curve")
    )

    pivot.index.name = "DepMap_ID"
    return pivot


###############################################################################
# PRISM Repurposing secondary screen (19Q4)
###############################################################################


def _load_prism_matrix(
    matrix_path: str,
    treatment_info_path: str,
    cell_info_path: str,
    model: pd.DataFrame,
) -> pd.DataFrame:
    """
    Load PRISM secondary screen replicate-collapsed logfold-change and convert
    into a DepMap_ID x Compound matrix.

    Steps:
      - matrix: rows are PRISM cell IDs, columns are PRISM treatment IDs
      - cell info: map PRISM cell ID -> CCLE/DepMap_ID (varies by version)
      - treatment info: map treatment ID -> compound name and concentration
      - for each compound, average across doses for a given cell line
    """
    # Load matrix - first column (unnamed) contains PRISM cell IDs as row names
    prism_matrix = pd.read_csv(matrix_path, index_col=0, low_memory=False)
    # Index contains PRISM cell IDs like "PR500_ACH-000007"
    prism_matrix.index.name = "prism_cell_id"

    # Load cell info
    cell_info = pd.read_csv(cell_info_path, low_memory=False)
    cell_info.columns = [c.lower() for c in cell_info.columns]

    model_tmp = model.copy()
    model_tmp.columns = [c.lower() for c in model_tmp.columns]

    # Try to map via DepMap_ID (ModelID) directly or via CCLE name
    depmap_col = None
    if "depmap_id" in cell_info.columns:
        depmap_col = "depmap_id"
    elif "modelid" in cell_info.columns:
        # ModelID is the same as DepMap_ID, rename for consistency
        cell_info["depmap_id"] = cell_info["modelid"]
        depmap_col = "depmap_id"
    elif "ccle_name" in cell_info.columns:
        if "ccle_name" not in model_tmp.columns or "modelid" not in model_tmp.columns:
            raise ValueError(
                "Model.csv missing columns needed to map PRISM CCLE_Name to ModelID (DepMap_ID)"
            )
        ccle_to_depmap = (
            model_tmp[["ccle_name", "modelid"]]
            .dropna()
            .drop_duplicates()
            .set_index("ccle_name")["modelid"]
        )
        cell_info["depmap_id"] = cell_info["ccle_name"].map(ccle_to_depmap)
        depmap_col = "depmap_id"
    else:
        raise ValueError(
            "PRISM cell info must contain 'DepMap_ID' or 'CCLE_Name' column"
        )

    # Map PRISM cell IDs (row_name) to DepMap_ID
    # The cell_info file has 'row_name' column with PRISM IDs like "PR500_ACH-000824"
    if "row_name" not in cell_info.columns:
        raise ValueError(
            "PRISM cell info must contain 'row_name' column with PRISM cell IDs"
        )

    cell_map = (
        cell_info[["row_name", depmap_col]]
        .dropna()
        .drop_duplicates()
        .rename(columns={"row_name": "prism_cell_id", depmap_col: "DepMap_ID"})
        .set_index("prism_cell_id")["DepMap_ID"]
    )

    # Subset matrix to cells that map to DepMap_ID
    # prism_matrix index contains PRISM cell IDs
    valid_cells = prism_matrix.index[prism_matrix.index.isin(cell_map.index)]
    prism_matrix = prism_matrix.loc[valid_cells].copy()
    prism_matrix["DepMap_ID"] = prism_matrix.index.map(cell_map)
    prism_matrix = prism_matrix.reset_index(drop=True)

    # Now load treatment info (treatment ID -> compound name)
    tinfo = pd.read_csv(treatment_info_path, low_memory=False)
    tinfo.columns = [c.lower() for c in tinfo.columns]

    # Treatment ID is in 'column_name' column (e.g., "BRD-K36788280-001-01-2::0.15625::MTS010::PROS001")
    if "column_name" not in tinfo.columns:
        raise ValueError("PRISM treatment info missing 'column_name' column")

    # Pick a human-readable compound label
    if "name" in tinfo.columns:
        name_col = "name"
    elif "compound" in tinfo.columns:
        name_col = "compound"
    else:
        # Fallback to broad_id if available, otherwise column_name
        if "broad_id" in tinfo.columns:
            name_col = "broad_id"
        else:
            name_col = "column_name"

    # Some versions have multiple doses per compound; collapse to mean per compound
    tinfo["compound_label"] = tinfo[name_col].astype(str)
    # Add BRD ID if available for better compound identification
    if "broad_id" in tinfo.columns and name_col != "broad_id":
        tinfo["compound_label"] = (
            tinfo["compound_label"] + " (BRD:" + tinfo["broad_id"].astype(str) + ")"
        )

    # Build treatment_id (column_name) -> compound_label map
    treatment_to_compound = (
        tinfo[["column_name", "compound_label"]]
        .dropna()
        .drop_duplicates()
        .set_index("column_name")["compound_label"]
    )

    # The matrix columns are treatment IDs; rename them to compound labels
    # First, melt to long format
    value_cols = [c for c in prism_matrix.columns if c != "DepMap_ID"]
    long = prism_matrix.melt(
        id_vars="DepMap_ID",
        value_vars=value_cols,
        var_name="treatment_id",
        value_name="logfold_change",
    )

    long = long[long["treatment_id"].isin(treatment_to_compound.index)].copy()
    long["compound_label"] = long["treatment_id"].map(treatment_to_compound)

    # Average across doses per compound
    agg = (
        long.groupby(["DepMap_ID", "compound_label"])["logfold_change"]
        .mean()
        .reset_index()
    )

    prism_matrix_compound = agg.pivot(
        index="DepMap_ID", columns="compound_label", values="logfold_change"
    )
    prism_matrix_compound.index.name = "DepMap_ID"

    return prism_matrix_compound


def _load_prism_public_24q2(
    matrix_path: str,
    treatment_info_path: str,
    cell_info_path: str,
    model: pd.DataFrame,
) -> pd.DataFrame:
    """
    Load PRISM Repurposing Public 24Q2 data and convert into a DepMap_ID x Compound matrix.
    
    The matrix file is in long format with columns:
    - row_id: cell line identifier (e.g., "ACH-000001::P946.2::PR500B::REP300")
    - broad_id: compound BRD ID
    - dose: dose level
    - LFC: log fold change value
    
    Steps:
    1. Parse row_id to extract DepMap_ID (first part before "::")
    2. Join with treatment metadata to get compound names
    3. Average across doses per compound per cell line
    4. Pivot to DepMap_ID x Compound matrix
    """
    # Load the long-format matrix
    df = pd.read_csv(matrix_path, low_memory=False)
    
    # Extract DepMap_ID from row_id (format: "ACH-000001::P946.2::PR500B::REP300")
    df["DepMap_ID"] = df["row_id"].str.split("::").str[0]
    
    # Load treatment metadata to get compound names
    tinfo = pd.read_csv(treatment_info_path, low_memory=False)
    tinfo.columns = [c.lower() for c in tinfo.columns]
    
    # Create compound label with BRD ID
    if "name" in tinfo.columns:
        name_col = "name"
    else:
        name_col = "broad_id"
    
    tinfo["compound_label"] = tinfo[name_col].astype(str)
    if "broad_id" in tinfo.columns:
        tinfo["compound_label"] = (
            tinfo["compound_label"] + " (BRD:" + tinfo["broad_id"].astype(str) + ")"
        )
    
    # Map broad_id to compound_label
    # Handle duplicates by taking the first occurrence
    broad_to_compound = (
        tinfo[["broad_id", "compound_label"]]
        .dropna()
        .drop_duplicates(subset=["broad_id"], keep="first")
        .set_index("broad_id")["compound_label"]
    )
    
    # Map broad_id to compound name
    df["compound_label"] = df["broad_id"].map(broad_to_compound)
    df = df[df["compound_label"].notna()].copy()
    
    # Average across doses per compound per cell line
    agg = (
        df.groupby(["DepMap_ID", "compound_label"])["LFC"]
        .mean()
        .reset_index()
    )
    
    # Pivot to DepMap_ID x Compound matrix
    pivot = agg.pivot(index="DepMap_ID", columns="compound_label", values="LFC")
    pivot = pivot.fillna(np.nan)  # Ensure NaN for missing values
    
    return pivot


###############################################################################
# Association computation
###############################################################################


def _compute_correlations(
    expression: pd.DataFrame,
    drug_matrix: pd.DataFrame,
    dataset_label: str,
) -> pd.DataFrame:
    """
    Compute Pearson correlations between each gene in `expression` and each
    drug in `drug_matrix`.

    Both DataFrames must be indexed by DepMap_ID.
    """
    # Align on common DepMap_IDs
    # Ensure both dataframes have unique indices (handle duplicates by taking mean)
    if expression.index.duplicated().any():
        expression = expression.groupby(expression.index).mean()
    if drug_matrix.index.duplicated().any():
        drug_matrix = drug_matrix.groupby(drug_matrix.index).mean()
    
    # Get unique indices
    expr_idx = expression.index
    drug_idx = drug_matrix.index
    
    common_ids = expr_idx.intersection(drug_idx)
    if len(common_ids) == 0:
        raise ValueError(f"No overlapping DepMap_IDs between expression and {dataset_label}")

    # Sort for consistent ordering
    common_ids = common_ids.sort_values() if hasattr(common_ids, 'sort_values') else sorted(common_ids)
    
    # Use loc to select common IDs (more reliable than reindex when indices are already unique)
    expr_aligned = expression.loc[common_ids]
    drug_aligned = drug_matrix.loc[common_ids]
    
    # Drop rows where either expression or drug data is completely missing
    expr_has_data = expr_aligned.notna().any(axis=1)
    drug_has_data = drug_aligned.notna().any(axis=1)
    valid_rows = expr_has_data & drug_has_data
    
    expr_aligned = expr_aligned[valid_rows]
    drug_aligned = drug_aligned[valid_rows]
    
    # Verify alignment - both should have the same number of rows
    if len(expr_aligned) != len(drug_aligned):
        raise ValueError(
            f"Alignment failed: expression has {len(expr_aligned)} rows, "
            f"drug_matrix has {len(drug_aligned)} rows after alignment"
        )
    
    # Final check: ensure indices match exactly
    if not expr_aligned.index.equals(drug_aligned.index):
        # If indices don't match, force alignment by creating new index
        aligned_idx = expr_aligned.index.intersection(drug_aligned.index)
        expr_aligned = expr_aligned.loc[aligned_idx]
        drug_aligned = drug_aligned.loc[aligned_idx]

    results: List[Tuple[str, str, float]] = []

    # Pre-convert to numpy for speed
    expr_values = expr_aligned.to_numpy()
    drug_values = drug_aligned.to_numpy()
    
    # Final safety check
    if expr_values.shape[0] != drug_values.shape[0]:
        raise ValueError(
            f"Shape mismatch after conversion: expression has {expr_values.shape[0]} rows, "
            f"drug_matrix has {drug_values.shape[0]} rows"
        )

    gene_names = list(expr_aligned.columns)
    drug_names = list(drug_aligned.columns)

    # Center columns (mean 0) to make correlation computation faster
    expr_values_centered = expr_values - np.nanmean(expr_values, axis=0, keepdims=True)
    drug_values_centered = drug_values - np.nanmean(drug_values, axis=0, keepdims=True)

    # For each gene / drug pair, compute Pearson r dropping NaNs
    for gi, gene in enumerate(gene_names):
        gvec = expr_values_centered[:, gi]
        for dj, drug in enumerate(drug_names):
            dvec = drug_values_centered[:, dj]
            # Ensure vectors have the same length
            if len(gvec) != len(dvec):
                continue
            mask = np.isfinite(gvec) & np.isfinite(dvec)
            if mask.sum() < 3:
                continue
            gv = gvec[mask]
            dv = dvec[mask]

            # Pearson correlation
            denom = np.sqrt(np.sum(gv ** 2) * np.sum(dv ** 2))
            if denom == 0:
                continue
            r = float(np.sum(gv * dv) / denom)
            results.append((gene, drug, r))

    if not results:
        return pd.DataFrame(columns=["Gene/Compound", "Dataset", "Correlation", "other_entity_type"])

    out = pd.DataFrame(results, columns=["Gene", "Compound", "Correlation"])
    out["Dataset"] = dataset_label
    out["Gene/Compound"] = out["Compound"]
    out["other_entity_type"] = "compound_experiment"  # All drug associations are compound experiments
    
    # Round correlation to 3 decimal places to match DepMap format
    out["Correlation"] = out["Correlation"].round(3)

    # Order similar to DepMap portal: sort by absolute correlation (descending)
    out = out.sort_values("Correlation", key=lambda s: s.abs(), ascending=False)

    # Return columns in the same order as DepMap: Gene/Compound, Dataset, Correlation, other_entity_type
    return out[["Gene/Compound", "Dataset", "Correlation", "other_entity_type"]]


###############################################################################
# Command-line interface
###############################################################################


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute gene–drug sensitivity associations similar to DepMap's "
            "precomputed associations (expression vs drug sensitivity)."
        )
    )
    parser.add_argument(
        "--genes",
        required=True,
        help="Comma-separated list of HGNC gene symbols (e.g. EGFR,BRAF,KRAS)",
    )
    parser.add_argument(
        "--expression-file",
        default=DEFAULT_EXPRESSION_FILE,
        help=f"Expression matrix CSV (default: {DEFAULT_EXPRESSION_FILE})",
    )
    parser.add_argument(
        "--model-file",
        default=DEFAULT_MODEL_FILE,
        help=f"DepMap Model.csv file (default: {DEFAULT_MODEL_FILE})",
    )
    parser.add_argument(
        "--gdsc-file",
        default=DEFAULT_GDSC_DOSE_RESPONSE_FILE,
        help=f"Sanger GDSC dose-response CSV (default: {DEFAULT_GDSC_DOSE_RESPONSE_FILE})",
    )
    parser.add_argument(
        "--ctrp-zip",
        default=DEFAULT_CTRP_ZIP_FILE,
        help=f"CTRP expanded dataset ZIP (default: {DEFAULT_CTRP_ZIP_FILE})",
    )
    parser.add_argument(
        "--prism-secondary-matrix",
        default=DEFAULT_PRISM_SECONDARY_MATRIX_FILE,
        help=f"PRISM Secondary Screen matrix CSV (default: {DEFAULT_PRISM_SECONDARY_MATRIX_FILE})",
    )
    parser.add_argument(
        "--prism-secondary-treatment-info",
        default=DEFAULT_PRISM_SECONDARY_TREATMENT_INFO_FILE,
        help=f"PRISM Secondary Screen treatment info CSV (default: {DEFAULT_PRISM_SECONDARY_TREATMENT_INFO_FILE})",
    )
    parser.add_argument(
        "--prism-secondary-cell-info",
        default=DEFAULT_PRISM_SECONDARY_CELL_INFO_FILE,
        help=f"PRISM Secondary Screen cell line info CSV (default: {DEFAULT_PRISM_SECONDARY_CELL_INFO_FILE})",
    )
    parser.add_argument(
        "--prism-public-matrix",
        default=DEFAULT_PRISM_PUBLIC_MATRIX_FILE,
        help=f"PRISM Public 24Q2 matrix CSV (default: {DEFAULT_PRISM_PUBLIC_MATRIX_FILE})",
    )
    parser.add_argument(
        "--prism-public-treatment-info",
        default=DEFAULT_PRISM_PUBLIC_TREATMENT_INFO_FILE,
        help=f"PRISM Public 24Q2 treatment info CSV (default: {DEFAULT_PRISM_PUBLIC_TREATMENT_INFO_FILE})",
    )
    parser.add_argument(
        "--prism-public-cell-info",
        default=DEFAULT_PRISM_PUBLIC_CELL_INFO_FILE,
        help=f"PRISM Public 24Q2 cell line info CSV (default: {DEFAULT_PRISM_PUBLIC_CELL_INFO_FILE})",
    )
    parser.add_argument(
        "--no-gdsc",
        action="store_true",
        help="Disable GDSC drug sensitivity associations",
    )
    parser.add_argument(
        "--no-ctrp",
        action="store_true",
        help="Disable CTRP drug sensitivity associations",
    )
    parser.add_argument(
        "--no-prism",
        action="store_true",
        help="Disable PRISM drug sensitivity associations",
    )
    parser.add_argument(
        "--output",
        default="associations_output.csv",
        help="Path to write the combined associations CSV",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    genes = [g.strip() for g in args.genes.split(",") if g.strip()]

    if not genes:
        raise SystemExit("No valid genes provided via --genes")

    # Sanity-check existence of core files
    for path, label in [
        (args.expression_file, "expression"),
        (args.model_file, "Model.csv"),
    ]:
        if not os.path.exists(path):
            raise SystemExit(f"Required {label} file not found: {path}")

    print(f"Loading expression for genes: {', '.join(genes)}")
    expression = _load_expression(args.expression_file, genes)

    print("Loading model metadata (Model.csv)")
    model = _load_model_metadata(args.model_file)

    all_results: List[pd.DataFrame] = []

    if not args.no_gdsc and os.path.exists(args.gdsc_file):
        print("Loading GDSC dose-response data...")
        gdsc_datasets = _load_gdsc_auc(args.gdsc_file, model)
        print(f"Found {len(gdsc_datasets)} GDSC dataset(s)")
        for gdsc, dataset_label in gdsc_datasets:
            print(f"Computing expression–{dataset_label} associations...")
            gdsc_res = _compute_correlations(expression, gdsc, dataset_label=dataset_label)
            all_results.append(gdsc_res)
    else:
        print("Skipping GDSC (file missing or disabled)")

    if not args.no_ctrp and os.path.exists(args.ctrp_zip):
        print("Loading CTRP curves (area_under_curve) from ZIP...")
        ctrp = _load_ctrp_auc(args.ctrp_zip, model)
        print("Computing expression–CTRP associations...")
        ctrp_res = _compute_correlations(
            expression, ctrp, dataset_label="Drug sensitivity AUC (CTD^2)"
        )
        all_results.append(ctrp_res)
    else:
        print("Skipping CTRP (file missing or disabled)")

    # PRISM Secondary Screen
    if (
        not args.no_prism
        and os.path.exists(args.prism_secondary_matrix)
        and os.path.exists(args.prism_secondary_treatment_info)
        and os.path.exists(args.prism_secondary_cell_info)
    ):
        print("Loading PRISM Secondary Screen data...")
        prism_secondary = _load_prism_matrix(
            args.prism_secondary_matrix,
            args.prism_secondary_treatment_info,
            args.prism_secondary_cell_info,
            model,
        )
        print("Computing expression–PRISM Secondary Screen associations...")
        prism_secondary_res = _compute_correlations(
            expression,
            prism_secondary,
            dataset_label="Drug sensitivity AUC (PRISM Repurposing Secondary Screen)",
        )
        all_results.append(prism_secondary_res)
    else:
        print("Skipping PRISM Secondary Screen (files missing or disabled)")
    
    # PRISM Public 24Q2
    if (
        not args.no_prism
        and os.path.exists(args.prism_public_matrix)
        and os.path.exists(args.prism_public_treatment_info)
        and os.path.exists(args.prism_public_cell_info)
    ):
        print("Loading PRISM Public 24Q2 data...")
        prism_public = _load_prism_public_24q2(
            args.prism_public_matrix,
            args.prism_public_treatment_info,
            args.prism_public_cell_info,
            model,
        )
        print("Computing expression–PRISM Public associations...")
        prism_public_res = _compute_correlations(
            expression,
            prism_public,
            dataset_label="PRISM Repurposing Public 24Q2",  # TODO: Update version here when newer becomes available (e.g., "PRISM Repurposing Public 25Q3")  # TODO: Update version here when newer becomes available (e.g., "PRISM Repurposing Public 25Q3")
        )
        all_results.append(prism_public_res)
    else:
        print("Skipping PRISM Public 24Q2 (files missing or disabled)")

    if not all_results:
        raise SystemExit("No drug sensitivity datasets were loaded; nothing to compute.")

    combined = pd.concat(all_results, ignore_index=True)

    # Sort as in DepMap's precomputed associations: absolute correlation descending
    combined = combined.sort_values(
        "Correlation", key=lambda s: s.abs(), ascending=False
    )

    combined.to_csv(args.output, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"Wrote associations for {len(genes)} gene(s) to {args.output}")


if __name__ == "__main__":
    main()


