import { NextRequest, NextResponse } from "next/server";

/**
 * Aggregates molecular metadata from PubChem (compound SMILES) and RCSB PDB
 * (gene-associated structures and ligand SMILES from chemical components).
 *
 * PubChem PUG REST: https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest
 * RCSB Search API v2: https://search.rcsb.org/#graphql-overview
 * RCSB Data API: https://data.rcsb.org/#rest-apis
 */

/** PubChem PUG may return `SMILES` / `ConnectivitySMILES` when CanonicalSMILES was requested. */
type PubChemPropsRow = {
  CID: number;
  MolecularFormula?: string;
  MolecularWeight?: string | number;
  CanonicalSMILES?: string;
  IsomericSMILES?: string;
  SMILES?: string;
  ConnectivitySMILES?: string;
  IUPACName?: string;
  XLogP?: number;
  TPSA?: number;
  Complexity?: number;
  HBondDonorCount?: number;
  HBondAcceptorCount?: number;
  RotatableBondCount?: number;
  ExactMass?: string;
  MonoisotopicMass?: string;
};

type PubChemPropertyResponse = {
  PropertyTable?: {
    Properties?: PubChemPropsRow[];
  };
};

/** Extended properties requested in one PUG call (see PubChem PUG REST Compound Property). */
const PUBCHEM_PROPERTIES =
  "CanonicalSMILES,IsomericSMILES,MolecularFormula,MolecularWeight,IUPACName,XLogP,TPSA,Complexity,HBondDonorCount,HBondAcceptorCount,RotatableBondCount,ExactMass,MonoisotopicMass";

function pickPubChemSmiles(p: PubChemPropsRow): {
  canonicalSmiles: string | null;
  isomericSmiles: string | null;
} {
  const canonicalSmiles =
    p.CanonicalSMILES ??
    p.SMILES ??
    p.ConnectivitySMILES ??
    null;
  const isomericSmiles = p.IsomericSMILES ?? null;
  return { canonicalSmiles, isomericSmiles };
}

type RcsbSearchResponse = {
  result_set?: Array<{ identifier: string }>;
  total_count?: number;
};

type RcsbEntryCore = {
  rcsb_entry_container_identifiers?: { entry_id?: string };
  rcsb_entry_info?: {
    nonpolymer_bound_components?: string[];
  };
};

type ChemCompResponse = {
  chem_comp?: { id?: string; name?: string };
  rcsb_chem_comp_descriptor?: {
    SMILES?: string;
    SMILES_stereo?: string;
  };
};

/**
 * DepMap association CSV often labels compounds like "BI-2536 (GDSC2:1086)".
 * PubChem expects the drug name only — strip trailing "(SOURCE: id)" annotations.
 */
function normalizeDepmapCompoundLabel(raw: string): string {
  let s = raw.trim();
  // Remove nested trailing parentheses: "… (GDSC2:1086)" → "…"
  while (/\s*\([^)]*\)\s*$/.test(s)) {
    s = s.replace(/\s*\([^)]*\)\s*$/, "").trim();
  }
  return s;
}

/** Strip DepMap suffixes from gene labels; normalize to uppercase HGNC-style symbol for lookup. */
function normalizeDepmapGeneLabel(raw: string): string {
  let s = raw.trim();
  while (/\s*\([^)]*\)\s*$/.test(s)) {
    s = s.replace(/\s*\([^)]*\)\s*$/, "").trim();
  }
  return s.toUpperCase();
}

type RcsbGeneStrategy = "uniprot_accession_match" | "full_text_gene";

/**
 * Resolve a human gene symbol or UniProt accession to a UniProt accession.
 * RCSB no longer enables text search on rcsb_polymer_entity.rcsb_gene_name — we map via UniProt.
 */
async function resolveHumanUniprotAccession(query: string): Promise<string | null> {
  const q = query.trim().toUpperCase();
  if (!q) return null;

  const direct = await fetch(
    `https://rest.uniprot.org/uniprotkb/${encodeURIComponent(q)}?fields=accession&format=json`,
    { next: { revalidate: 86400 } }
  );
  if (direct.ok) {
    const j = (await direct.json()) as { primaryAccession?: string };
    if (j.primaryAccession) return j.primaryAccession;
  }

  const searchUrl =
    `https://rest.uniprot.org/uniprotkb/search?query=` +
    encodeURIComponent(`(gene:${q}) AND (organism_id:9606)`) +
    `&fields=accession&format=json&size=10`;
  const res = await fetch(searchUrl, { next: { revalidate: 86400 } });
  if (!res.ok) return null;
  const data = (await res.json()) as {
    results?: Array<{ primaryAccession?: string; entryType?: string }>;
  };
  const results = data.results ?? [];
  const reviewed = results.find((r) => r.entryType?.includes("reviewed"));
  return (reviewed ?? results[0])?.primaryAccession ?? null;
}

async function rcsbSearchByUniprotAccession(accession: string): Promise<string[]> {
  const body = {
    query: {
      type: "terminal",
      service: "text",
      parameters: {
        attribute:
          "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession",
        operator: "exact_match",
        value: accession,
      },
    },
    return_type: "entry",
    request_options: {
      paginate: { start: 0, rows: 15 },
      results_content_type: ["experimental", "computational"],
      sort: [
        {
          sort_by: "rcsb_accession_info.initial_release_date",
          direction: "desc" as const,
        },
      ],
    },
  };

  const res = await fetch("https://search.rcsb.org/rcsbsearch/v2/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    next: { revalidate: 3600 },
  });
  if (!res.ok) return [];
  const data = (await res.json()) as RcsbSearchResponse;
  return data.result_set?.map((r) => r.identifier).filter(Boolean) ?? [];
}

async function rcsbFullTextGeneSymbol(symbol: string): Promise<string[]> {
  const body = {
    query: {
      type: "terminal",
      service: "full_text",
      parameters: { value: symbol.trim().toUpperCase() },
    },
    return_type: "entry",
    request_options: {
      paginate: { start: 0, rows: 15 },
      results_content_type: ["experimental", "computational"],
      sort: [
        {
          sort_by: "rcsb_accession_info.initial_release_date",
          direction: "desc" as const,
        },
      ],
    },
  };

  const res = await fetch("https://search.rcsb.org/rcsbsearch/v2/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    next: { revalidate: 3600 },
  });
  if (!res.ok) return [];
  const data = (await res.json()) as RcsbSearchResponse;
  return data.result_set?.map((r) => r.identifier).filter(Boolean) ?? [];
}

async function searchRcsbForGene(geneQuery: string): Promise<{
  pdbIds: string[];
  strategy: RcsbGeneStrategy | null;
  uniprotAccession: string | null;
}> {
  const accession = await resolveHumanUniprotAccession(geneQuery);
  if (accession) {
    const ids = await rcsbSearchByUniprotAccession(accession);
    if (ids.length > 0) {
      return {
        pdbIds: ids,
        strategy: "uniprot_accession_match",
        uniprotAccession: accession,
      };
    }
  }

  const idsFt = await rcsbFullTextGeneSymbol(geneQuery);
  if (idsFt.length > 0) {
    return {
      pdbIds: idsFt,
      strategy: "full_text_gene",
      uniprotAccession: accession,
    };
  }

  return { pdbIds: [], strategy: null, uniprotAccession: accession };
}

async function fetchPubChem(compoundName: string) {
  const base = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound";
  const nameEnc = encodeURIComponent(compoundName.trim());

  const propUrl = `${base}/name/${nameEnc}/property/${PUBCHEM_PROPERTIES}/JSON`;
  let res = await fetch(propUrl, { next: { revalidate: 3600 } });

  if (!res.ok) {
    const cidsUrl = `${base}/name/${nameEnc}/cids/JSON`;
    const cidsRes = await fetch(cidsUrl, { next: { revalidate: 3600 } });
    if (!cidsRes.ok) {
      return {
        ok: false as const,
        error: `PubChem: could not resolve name "${compoundName}" (${res.status})`,
      };
    }
    const cidsData = (await cidsRes.json()) as {
      IdentifierList?: { CID?: number[] };
    };
    const cid = cidsData.IdentifierList?.CID?.[0];
    if (!cid) {
      return {
        ok: false as const,
        error: `PubChem: no CID found for "${compoundName}"`,
      };
    }
    const pUrl = `${base}/cid/${cid}/property/${PUBCHEM_PROPERTIES}/JSON`;
    res = await fetch(pUrl, { next: { revalidate: 3600 } });
    if (!res.ok) {
      return {
        ok: false as const,
        error: `PubChem: property fetch failed for CID ${cid}`,
      };
    }
  }

  const data = (await res.json()) as PubChemPropertyResponse;
  const p = data.PropertyTable?.Properties?.[0];
  if (!p?.CID) {
    return {
      ok: false as const,
      error: `PubChem: no properties returned for "${compoundName}"`,
    };
  }

  const { canonicalSmiles, isomericSmiles } = pickPubChemSmiles(p);

  return {
    ok: true as const,
    cid: p.CID,
    canonicalSmiles,
    isomericSmiles,
    molecularFormula: p.MolecularFormula ?? null,
    molecularWeight: p.MolecularWeight ?? null,
    iupacName: p.IUPACName ?? null,
    xlogp: p.XLogP ?? null,
    tpsa: p.TPSA ?? null,
    complexity: p.Complexity ?? null,
    hBondDonorCount: p.HBondDonorCount ?? null,
    hBondAcceptorCount: p.HBondAcceptorCount ?? null,
    rotatableBondCount: p.RotatableBondCount ?? null,
    exactMass: p.ExactMass ?? null,
    monoisotopicMass: p.MonoisotopicMass ?? null,
    pubchemUrl: `https://pubchem.ncbi.nlm.nih.gov/compound/${p.CID}`,
  };
}

type UniprotGeneSummary = {
  accession: string;
  proteinName: string | null;
  geneSymbol: string | null;
  organism: string | null;
  sequenceLength: number | null;
  uniprotUrl: string;
  /** Single-letter amino acid sequence from UniProt (complete canonical protein). */
  sequence: string | null;
  /** Calculated molecular mass (Da), from UniProt sequence record. */
  sequenceMassDa: number | null;
  sequenceMd5: string | null;
};

async function fetchUniprotGeneSummary(
  accession: string | null
): Promise<UniprotGeneSummary | null> {
  if (!accession?.trim()) return null;
  const acc = accession.trim();
  const url =
    `https://rest.uniprot.org/uniprotkb/${encodeURIComponent(acc)}` +
    `?format=json&fields=gene_names,organism_name,sequence,protein_name`;
  const res = await fetch(url, { next: { revalidate: 86400 } });
  if (!res.ok) return null;
  const j = (await res.json()) as {
    primaryAccession?: string;
    organism?: { scientificName?: string };
    genes?: Array<{ geneName?: { value?: string } }>;
    sequence?: {
      value?: string;
      length?: number;
      molWeight?: number;
      md5?: string;
    };
    proteinDescription?: {
      recommendedName?: { fullName?: { value?: string } };
    };
  };
  const proteinName =
    j.proteinDescription?.recommendedName?.fullName?.value ?? null;
  const geneSymbol = j.genes?.[0]?.geneName?.value ?? null;
  const organism = j.organism?.scientificName ?? null;
  const seq = j.sequence;
  const sequenceLength = seq?.length ?? null;
  const sequence = seq?.value?.trim() ?? null;
  const sequenceMassDa =
    typeof seq?.molWeight === "number" ? seq.molWeight : null;
  const sequenceMd5 = seq?.md5 ?? null;
  return {
    accession: j.primaryAccession ?? acc,
    proteinName,
    geneSymbol,
    organism,
    sequenceLength,
    sequence,
    sequenceMassDa,
    sequenceMd5,
    uniprotUrl: `https://www.uniprot.org/uniprotkb/${encodeURIComponent(j.primaryAccession ?? acc)}`,
  };
}

type RcsbEntrySummary = {
  pdbId: string;
  title: string | null;
  experimentalMethod: string | null;
  resolutionAngstrom: number | null;
};

async function fetchRcsbEntrySummary(
  pdbId: string | null
): Promise<RcsbEntrySummary | null> {
  if (!pdbId?.trim()) return null;
  const id = pdbId.trim().toUpperCase();
  const url = `https://data.rcsb.org/rest/v1/core/entry/${encodeURIComponent(id)}`;
  const res = await fetch(url, { next: { revalidate: 3600 } });
  if (!res.ok) return null;
  const e = (await res.json()) as {
    struct?: { title?: string };
    exptl?: Array<{ method?: string }>;
    rcsb_entry_info?: {
      resolution_combined?: number[];
      experimental_method?: string;
    };
  };
  const title = e.struct?.title ?? null;
  const experimentalMethod =
    e.exptl?.[0]?.method ?? e.rcsb_entry_info?.experimental_method ?? null;
  const rc = e.rcsb_entry_info?.resolution_combined;
  const resolutionAngstrom =
    Array.isArray(rc) && rc.length > 0 && typeof rc[0] === "number"
      ? rc[0]
      : null;
  return {
    pdbId: id,
    title,
    experimentalMethod,
    resolutionAngstrom,
  };
}

async function fetchEntryLigandInfo(pdbId: string) {
  const url = `https://data.rcsb.org/rest/v1/core/entry/${pdbId}`;
  const res = await fetch(url, { next: { revalidate: 3600 } });
  if (!res.ok) {
    return { pdbId, ligands: [] as Array<{ compId: string; name: string; smiles: string | null }> };
  }

  const entry = (await res.json()) as RcsbEntryCore;
  const comps =
    entry.rcsb_entry_info?.nonpolymer_bound_components?.filter(
      (c) => c && c !== "HOH"
    ) ?? [];

  const ligands: Array<{ compId: string; name: string; smiles: string | null }> =
    [];

  for (const compId of comps.slice(0, 5)) {
    const cr = await fetch(
      `https://data.rcsb.org/rest/v1/core/chemcomp/${encodeURIComponent(compId)}`,
      { next: { revalidate: 3600 } }
    );
    if (!cr.ok) continue;
    const cc = (await cr.json()) as ChemCompResponse;
    const smiles =
      cc.rcsb_chem_comp_descriptor?.SMILES ??
      cc.rcsb_chem_comp_descriptor?.SMILES_stereo ??
      null;
    ligands.push({
      compId,
      name: cc.chem_comp?.name ?? compId,
      smiles,
    });
  }

  return { pdbId, ligands };
}

export async function GET(req: NextRequest) {
  const compound = req.nextUrl.searchParams.get("compound");
  const gene = req.nextUrl.searchParams.get("gene");

  if (!compound?.trim() || !gene?.trim()) {
    return NextResponse.json(
      { error: "Query parameters compound and gene are required." },
      { status: 400 }
    );
  }

  const compoundRaw = compound.trim();
  const geneRaw = gene.trim();
  const manualPubchem = req.nextUrl.searchParams.get("pubchemName")?.trim();
  const manualGeneSearch = req.nextUrl.searchParams.get("geneSearchName")?.trim();

  /** Manual override wins; otherwise strip trailing "(GDSC2:…)" etc. from DepMap labels. */
  const compoundQuery =
    manualPubchem && manualPubchem.length > 0
      ? manualPubchem
      : normalizeDepmapCompoundLabel(compoundRaw);

  const geneQuery =
    manualGeneSearch && manualGeneSearch.length > 0
      ? manualGeneSearch.toUpperCase()
      : normalizeDepmapGeneLabel(geneRaw);

  const [pubchem, rcsbGene] = await Promise.all([
    fetchPubChem(compoundQuery),
    searchRcsbForGene(geneQuery),
  ]);

  const pdbIds = rcsbGene.pdbIds;
  const primaryPdbId = pdbIds[0] ?? null;

  const [geneProduct, representativeStructure, rcsbLigandsResult] =
    await Promise.all([
      fetchUniprotGeneSummary(rcsbGene.uniprotAccession),
      fetchRcsbEntrySummary(primaryPdbId),
      primaryPdbId
        ? fetchEntryLigandInfo(primaryPdbId).catch(() => ({
            pdbId: primaryPdbId,
            ligands: [] as Array<{
              compId: string;
              name: string;
              smiles: string | null;
            }>,
          }))
        : Promise.resolve(null),
    ]);

  let rcsbLigands = rcsbLigandsResult;
  if (primaryPdbId && !rcsbLigands) {
    rcsbLigands = { pdbId: primaryPdbId, ligands: [] };
  }

  const firstLigandSmiles = rcsbLigands?.ligands.find((l) => l.smiles) ?? null;

  return NextResponse.json({
    /** Full label from DepMap row (e.g. includes dataset suffix). */
    compound: compoundRaw,
    /** Name actually passed to PubChem (auto-normalized or manual override). */
    compoundQuery,
    /** Whether `pubchemName` query param was used for PubChem. */
    pubchemLookupMode:
      manualPubchem && manualPubchem.length > 0 ? "manual" : "auto",
    /** Gene label from the DepMap row (unchanged casing). */
    gene: geneRaw,
    /** Symbol / accession string used for UniProt → RCSB and fallbacks. */
    geneQuery,
    geneLookupMode:
      manualGeneSearch && manualGeneSearch.length > 0 ? "manual" : "auto",
    /** Human protein summary from UniProt when accession was resolved. */
    geneProduct,
    pubchem,
    rcsb: {
      pdbIds,
      primaryPdbId,
      structureUrl: primaryPdbId
        ? `https://www.rcsb.org/structure/${primaryPdbId}`
        : null,
      searchUrl: `https://www.rcsb.org/search?q=${encodeURIComponent(geneQuery)}`,
      searchStrategy: rcsbGene.strategy,
      uniprotAccession: rcsbGene.uniprotAccession,
      representativeStructure,
      /** Protein structures use PDB entries; SMILES here are for bound small-molecule ligands in the entry, not the protein itself. */
      note:
        "Genes encode proteins (there is no single small-molecule SMILES for a gene). We show the UniProt protein record and a representative PDB structure; ligand SMILES below are for cofactors or drugs in that PDB entry.",
      ligands: rcsbLigands?.ligands ?? [],
      primaryLigand: firstLigandSmiles,
    },
  });
}
