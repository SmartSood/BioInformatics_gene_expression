"use client";

import { useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { Card } from "@repo/ui/card";
import {
  ArrowLeft,
  Loader,
  ExternalLink,
  Dna,
  FlaskConical,
  AlertCircle,
  Search,
  RotateCcw,
  Copy,
  Check,
} from "lucide-react";

/** UniProt-style display: 50 aa per line, blocks of 10, ruler marks end of each block (1-based index). */
function formatUniProtSequenceLines(sequence: string): Array<{
  ruler: string;
  aa: string;
}> {
  const AA_PER_LINE = 50;
  const BLOCK = 10;
  const rows: Array<{ ruler: string; aa: string }> = [];
  const clean = sequence.replace(/\s+/g, "").toUpperCase();
  for (let start = 0; start < clean.length; start += AA_PER_LINE) {
    const slice = clean.slice(start, start + AA_PER_LINE);
    const blocks: string[] = [];
    for (let i = 0; i < slice.length; i += BLOCK) {
      blocks.push(slice.slice(i, i + BLOCK));
    }
    const ruler = blocks
      .map((_, idx) => {
        const cumLen = blocks
          .slice(0, idx + 1)
          .reduce((acc, b) => acc + b.length, 0);
        const endPos = start + cumLen;
        return String(endPos).padStart(blocks[idx].length, " ");
      })
      .join(" ");
    const aa = blocks.join(" ");
    rows.push({ ruler, aa });
  }
  return rows;
}

type MolecularResponse = {
  compound: string;
  /** Normalized drug name used for PubChem (DepMap labels often include " (GDSC2:…)" etc.). */
  compoundQuery?: string;
  /** Whether the user supplied `pubchemName` vs automatic stripping. */
  pubchemLookupMode?: "auto" | "manual";
  gene: string;
  /** Symbol or accession passed to UniProt / RCSB (uppercase). */
  geneQuery?: string;
  geneLookupMode?: "auto" | "manual";
  geneProduct?: {
    accession: string;
    proteinName: string | null;
    geneSymbol: string | null;
    organism: string | null;
    sequenceLength: number | null;
    uniprotUrl: string;
    sequence?: string | null;
    sequenceMassDa?: number | null;
    sequenceMd5?: string | null;
  } | null;
  pubchem:
    | {
        ok: true;
        cid: number;
        canonicalSmiles: string | null;
        isomericSmiles: string | null;
        molecularFormula: string | null;
        molecularWeight?: string | number | null;
        iupacName?: string | null;
        xlogp?: number | null;
        tpsa?: number | null;
        complexity?: number | null;
        hBondDonorCount?: number | null;
        hBondAcceptorCount?: number | null;
        rotatableBondCount?: number | null;
        exactMass?: string | null;
        monoisotopicMass?: string | null;
        pubchemUrl: string;
      }
    | { ok: false; error: string };
  rcsb: {
    pdbIds: string[];
    primaryPdbId: string | null;
    structureUrl: string | null;
    searchUrl: string;
    note: string;
    ligands: Array<{ compId: string; name: string; smiles: string | null }>;
    primaryLigand: {
      compId: string;
      name: string;
      smiles: string | null;
    } | null;
    searchStrategy?: "uniprot_accession_match" | "full_text_gene" | null;
    uniprotAccession?: string | null;
    representativeStructure?: {
      pdbId: string;
      title: string | null;
      experimentalMethod: string | null;
      resolutionAngstrom: number | null;
    } | null;
  };
};

export default function DepmapCompoundDetailPage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const gene = searchParams.get("gene") || "";
  const compound = searchParams.get("compound") || "";
  const experimentId = searchParams.get("experimentId") || "";
  const dataset = searchParams.get("dataset") || "";
  const correlationParam = searchParams.get("correlation");

  const [data, setData] = useState<MolecularResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  /** Value in the manual PubChem name field (synced to successful `compoundQuery`). */
  const [manualInput, setManualInput] = useState("");
  /**
   * null = derive name from DepMap label (auto strip suffix).
   * non-null string = pass as `pubchemName` to the API (manual lookup).
   */
  const [pubchemOverride, setPubchemOverride] = useState<string | null>(null);
  const [geneManualInput, setGeneManualInput] = useState("");
  /**
   * null = derive gene lookup from DepMap label (strip suffix, uppercase).
   * non-null = pass as `geneSearchName` to the API.
   */
  const [geneSearchOverride, setGeneSearchOverride] = useState<string | null>(
    null
  );
  const [sequenceCopied, setSequenceCopied] = useState(false);

  useEffect(() => {
    setPubchemOverride(null);
    setGeneSearchOverride(null);
    setSequenceCopied(false);
  }, [gene, compound]);

  useEffect(() => {
    if (!gene.trim() || !compound.trim()) {
      setError("Missing gene or compound in URL.");
      setLoading(false);
      return;
    }

    let cancelled = false;

    async function load() {
      setLoading(true);
      setError(null);
      try {
        const qs = new URLSearchParams({
          gene: gene.trim(),
          compound: compound.trim(),
        });
        if (pubchemOverride !== null && pubchemOverride.trim() !== "") {
          qs.set("pubchemName", pubchemOverride.trim());
        }
        if (geneSearchOverride !== null && geneSearchOverride.trim() !== "") {
          qs.set("geneSearchName", geneSearchOverride.trim());
        }
        const res = await fetch(`/api/depmap/molecular?${qs.toString()}`);
        if (!res.ok) {
          const errBody = await res.json().catch(() => ({}));
          throw new Error(
            (errBody as { error?: string }).error || `HTTP ${res.status}`
          );
        }
        const json = (await res.json()) as MolecularResponse;
        if (!cancelled) {
          setData(json);
          // Keep input aligned with the name actually used for PubChem
          setManualInput(json.compoundQuery ?? "");
          setGeneManualInput(json.geneQuery ?? "");
        }
      } catch (e: unknown) {
        if (!cancelled) {
          setError(
            e instanceof Error ? e.message : "Failed to load molecular data."
          );
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    load();
    return () => {
      cancelled = true;
    };
  }, [gene, compound, pubchemOverride, geneSearchOverride]);

  function applyPubchemLookup() {
    const t = manualInput.trim();
    setPubchemOverride(t === "" ? null : t);
  }

  function resetPubchemToAuto() {
    setPubchemOverride(null);
  }

  function applyGeneLookup() {
    const t = geneManualInput.trim();
    setGeneSearchOverride(t === "" ? null : t);
  }

  function resetGeneToAuto() {
    setGeneSearchOverride(null);
  }

  const correlation =
    correlationParam !== null && correlationParam !== ""
      ? parseFloat(correlationParam)
      : null;

  return (
    <div className="h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 overflow-y-auto">
      <div className="max-w-7xl mx-auto p-8">
        <div className="mb-6">
          <button
            type="button"
            onClick={() => router.back()}
            className="inline-flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-lg bg-slate-800/50 text-slate-300 border border-slate-700/50 hover:bg-slate-700/50 transition-colors"
          >
            <ArrowLeft className="w-4 h-4" />
            Back
          </button>
        </div>

        <Card color="slate" className="mb-6">
          <div className="flex flex-col gap-2">
            <h1 className="text-2xl font-bold text-white">
              Compound · gene context
            </h1>
            <p className="text-slate-400">
              Association for gene{" "}
              <span className="text-teal-400 font-semibold">{gene || "—"}</span>
              {" · "}
              compound{" "}
              <span className="text-purple-300 font-semibold">
                {compound || "—"}
              </span>
            </p>
            {(dataset || experimentId) && (
              <p className="text-sm text-slate-500">
                {dataset && (
                  <>
                    Dataset: <span className="text-slate-400">{dataset}</span>
                  </>
                )}
                {dataset && experimentId && " · "}
                {experimentId && (
                  <>
                    Experiment:{" "}
                    <span className="text-slate-400 font-mono text-xs">
                      {experimentId}
                    </span>
                  </>
                )}
              </p>
            )}
            {correlation !== null && !Number.isNaN(correlation) && (
              <p className="text-sm">
                <span className="text-slate-500">Correlation: </span>
                <span
                  className={
                    correlation > 0 ? "text-emerald-400" : "text-red-400"
                  }
                >
                  {correlation > 0 ? "+" : ""}
                  {correlation.toFixed(4)}
                </span>
              </p>
            )}

            <div className="mt-4 pt-4 border-t border-slate-700/50 space-y-3">
              <div>
                <p className="text-xs text-slate-500 uppercase tracking-wide mb-1">
                  Full compound label (DepMap row)
                </p>
                <p className="text-sm font-mono text-slate-200 bg-slate-900/60 border border-slate-700/50 rounded-lg px-3 py-2 break-all">
                  {compound || "—"}
                </p>
              </div>

              <div>
                <p className="text-xs text-slate-500 uppercase tracking-wide mb-1">
                  Full gene label (DepMap row)
                </p>
                <p className="text-sm font-mono text-slate-200 bg-slate-900/60 border border-slate-700/50 rounded-lg px-3 py-2 break-all">
                  {gene || "—"}
                </p>
              </div>

              <div>
                <label
                  htmlFor="pubchem-manual-name"
                  className="block text-sm font-medium text-slate-300 mb-1"
                >
                  Drug name for PubChem
                </label>
                <p className="text-xs text-slate-500 mb-2">
                  Automatic mode strips suffixes like{" "}
                  <span className="font-mono text-slate-400">(GDSC2:1086)</span>.
                  If PubChem still fails or the name is complex, edit below and
                  search again—the value below is what we send to PubChem and
                  store as{" "}
                  <span className="font-mono text-slate-400">compoundQuery</span>{" "}
                  on success.
                </p>
                <div className="flex flex-col sm:flex-row gap-2">
                  <input
                    id="pubchem-manual-name"
                    type="text"
                    value={manualInput}
                    onChange={(e) => setManualInput(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") applyPubchemLookup();
                    }}
                    disabled={loading}
                    className="flex-1 px-4 py-2.5 rounded-lg bg-slate-700/50 border border-slate-600/50 text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-teal-500/50 font-mono text-sm"
                    placeholder="e.g. BI-2536"
                  />
                  <button
                    type="button"
                    onClick={applyPubchemLookup}
                    disabled={loading}
                    className="inline-flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg bg-teal-600/80 text-white border border-teal-500/50 hover:bg-teal-500/80 disabled:opacity-50 text-sm font-medium shrink-0"
                  >
                    <Search className="w-4 h-4" />
                    Search PubChem
                  </button>
                  <button
                    type="button"
                    onClick={resetPubchemToAuto}
                    disabled={loading || pubchemOverride === null}
                    title={
                      pubchemOverride === null
                        ? "Already using automatic name from label"
                        : "Discard manual name and strip DepMap suffix again"
                    }
                    className="inline-flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg bg-slate-700/50 text-slate-300 border border-slate-600/50 hover:bg-slate-600/50 disabled:opacity-40 text-sm shrink-0"
                  >
                    <RotateCcw className="w-4 h-4" />
                    Auto from label
                  </button>
                </div>
                {data?.compoundQuery !== undefined && (
                  <p className="text-xs text-slate-500 mt-2">
                    Active PubChem name:{" "}
                    <span className="font-mono text-slate-300">
                      {data.compoundQuery}
                    </span>
                    {data.pubchemLookupMode === "manual" ? (
                      <span className="text-purple-400"> · manual</span>
                    ) : (
                      <span className="text-slate-600"> · auto</span>
                    )}
                  </p>
                )}
              </div>

              <div>
                <label
                  htmlFor="rcsb-gene-lookup"
                  className="block text-sm font-medium text-slate-300 mb-1"
                >
                  Gene / accession for PDB (RCSB)
                </label>
                <p className="text-xs text-slate-500 mb-2">
                  Automatic mode strips trailing annotations and uppercases the
                  symbol. We resolve human genes to{" "}
                  <span className="font-mono text-slate-400">UniProt</span> then
                  match PDB entries by accession. You can override with another
                  symbol or a UniProt ID (e.g.{" "}
                  <span className="font-mono text-slate-400">P45954</span>).
                </p>
                <div className="flex flex-col sm:flex-row gap-2">
                  <input
                    id="rcsb-gene-lookup"
                    type="text"
                    value={geneManualInput}
                    onChange={(e) => setGeneManualInput(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") applyGeneLookup();
                    }}
                    disabled={loading}
                    className="flex-1 px-4 py-2.5 rounded-lg bg-slate-700/50 border border-slate-600/50 text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500/50 font-mono text-sm"
                    placeholder="e.g. ACADSB or P45954"
                  />
                  <button
                    type="button"
                    onClick={applyGeneLookup}
                    disabled={loading}
                    className="inline-flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg bg-blue-600/80 text-white border border-blue-500/50 hover:bg-blue-500/80 disabled:opacity-50 text-sm font-medium shrink-0"
                  >
                    <Search className="w-4 h-4" />
                    Search PDB
                  </button>
                  <button
                    type="button"
                    onClick={resetGeneToAuto}
                    disabled={loading || geneSearchOverride === null}
                    title={
                      geneSearchOverride === null
                        ? "Already using automatic gene from label"
                        : "Discard manual lookup and use stripped DepMap label"
                    }
                    className="inline-flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg bg-slate-700/50 text-slate-300 border border-slate-600/50 hover:bg-slate-600/50 disabled:opacity-40 text-sm shrink-0"
                  >
                    <RotateCcw className="w-4 h-4" />
                    Auto from label
                  </button>
                </div>
                {data?.geneQuery !== undefined && (
                  <p className="text-xs text-slate-500 mt-2">
                    Active lookup:{" "}
                    <span className="font-mono text-slate-300">
                      {data.geneQuery}
                    </span>
                    {data.geneLookupMode === "manual" ? (
                      <span className="text-blue-400"> · manual</span>
                    ) : (
                      <span className="text-slate-600"> · auto</span>
                    )}
                  </p>
                )}
              </div>
            </div>
          </div>
        </Card>

        {loading && (
          <Card color="slate" className="text-center py-16">
            <Loader className="w-12 h-12 text-teal-400 mx-auto mb-4 animate-spin" />
            <p className="text-slate-400">
              Loading PubChem and RCSB PDB data…
            </p>
          </Card>
        )}

        {error && !loading && (
          <Card color="slate">
            <div className="flex items-start gap-3 bg-red-500/10 border border-red-500/30 rounded-lg p-4">
              <AlertCircle className="w-5 h-5 text-red-400 shrink-0 mt-0.5" />
              <p className="text-red-300">{error}</p>
            </div>
          </Card>
        )}

        {!loading && data && (
          <div className="space-y-6">
            {/* PubChem — compound SMILES */}
            <Card
              color="slate"
              title="Compound (PubChem)"
              icon={<FlaskConical className="w-5 h-5 text-emerald-400" />}
              iconColor="emerald"
            >
              <div className="space-y-4">
                {data.compoundQuery && (
                  <p className="text-sm text-slate-500 bg-slate-900/50 border border-slate-700/50 rounded-lg px-3 py-2">
                    Lookup name:{" "}
                    <span className="text-teal-400 font-mono font-medium">
                      {data.compoundQuery}
                    </span>
                    {data.pubchemLookupMode === "manual" ? (
                      <span className="text-slate-500"> (manual)</span>
                    ) : data.compoundQuery !== data.compound ? (
                      <span className="text-slate-500">
                        {" "}
                        (auto-stripped from full label)
                      </span>
                    ) : (
                      <span className="text-slate-500"> (auto)</span>
                    )}
                  </p>
                )}
                <p className="text-sm text-slate-400">
                  Structure and physicochemical properties from{" "}
                  <a
                    href="https://pubchem.ncbi.nlm.nih.gov/"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-teal-400 hover:underline inline-flex items-center gap-1"
                  >
                    PubChem
                    <ExternalLink className="w-3 h-3" />
                  </a>{" "}
                  (PUG REST).
                </p>

                {data.pubchem.ok ? (
                  <>
                    <div className="flex flex-wrap gap-3 items-center">
                      <a
                        href={data.pubchem.pubchemUrl}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-emerald-500/15 text-emerald-300 border border-emerald-500/40 hover:bg-emerald-500/25 text-sm font-medium"
                      >
                        Open compound on PubChem (CID {data.pubchem.cid})
                        <ExternalLink className="w-4 h-4" />
                      </a>
                    </div>

                    {(data.pubchem.molecularFormula ||
                      data.pubchem.molecularWeight != null ||
                      data.pubchem.iupacName) && (
                      <div className="rounded-lg border border-slate-700/50 bg-slate-900/40 p-4 space-y-2">
                        <div className="text-xs font-medium text-slate-500 uppercase tracking-wide">
                          Identity & formula
                        </div>
                        <dl className="grid grid-cols-1 sm:grid-cols-2 gap-x-6 gap-y-2 text-sm">
                          {data.pubchem.molecularFormula && (
                            <>
                              <dt className="text-slate-500">Molecular formula</dt>
                              <dd className="text-white font-mono">
                                {data.pubchem.molecularFormula}
                              </dd>
                            </>
                          )}
                          {data.pubchem.molecularWeight != null &&
                            data.pubchem.molecularWeight !== "" && (
                              <>
                                <dt className="text-slate-500">
                                  Molecular weight
                                </dt>
                                <dd className="text-slate-200">
                                  {String(data.pubchem.molecularWeight)}{" "}
                                  <span className="text-slate-500">Da</span>
                                </dd>
                              </>
                            )}
                          {data.pubchem.iupacName && (
                            <>
                              <dt className="text-slate-500 sm:col-span-1">
                                IUPAC name
                              </dt>
                              <dd className="text-slate-300 sm:col-span-1 break-words">
                                {data.pubchem.iupacName}
                              </dd>
                            </>
                          )}
                        </dl>
                      </div>
                    )}

                    {(data.pubchem.canonicalSmiles ||
                      data.pubchem.isomericSmiles) && (
                      <div className="space-y-3">
                        <div className="text-xs font-medium text-slate-500 uppercase tracking-wide">
                          SMILES
                        </div>
                        {data.pubchem.canonicalSmiles && (
                          <div>
                            <div className="text-xs text-slate-500 mb-1">
                              Canonical / connectivity (PubChem)
                            </div>
                            <pre className="text-sm text-slate-200 bg-slate-900/80 border border-slate-700/60 rounded-lg p-4 overflow-x-auto whitespace-pre-wrap break-all font-mono">
                              {data.pubchem.canonicalSmiles}
                            </pre>
                          </div>
                        )}
                        {data.pubchem.isomericSmiles &&
                          data.pubchem.isomericSmiles !==
                            data.pubchem.canonicalSmiles && (
                            <div>
                              <div className="text-xs text-slate-500 mb-1">
                                Isomeric SMILES (stereo)
                              </div>
                              <pre className="text-sm text-slate-200 bg-slate-900/80 border border-slate-700/60 rounded-lg p-4 overflow-x-auto whitespace-pre-wrap break-all font-mono">
                                {data.pubchem.isomericSmiles}
                              </pre>
                            </div>
                          )}
                      </div>
                    )}

                    {(data.pubchem.xlogp != null ||
                      data.pubchem.tpsa != null ||
                      data.pubchem.complexity != null ||
                      data.pubchem.hBondDonorCount != null ||
                      data.pubchem.hBondAcceptorCount != null ||
                      data.pubchem.rotatableBondCount != null ||
                      data.pubchem.exactMass != null) && (
                      <div className="rounded-lg border border-slate-700/50 bg-slate-900/40 p-4">
                        <div className="text-xs font-medium text-slate-500 uppercase tracking-wide mb-3">
                          Descriptors
                        </div>
                        <dl className="grid grid-cols-1 sm:grid-cols-2 gap-x-6 gap-y-2 text-sm">
                          {data.pubchem.xlogp != null && (
                            <>
                              <dt className="text-slate-500">XLogP</dt>
                              <dd className="text-slate-200">{data.pubchem.xlogp}</dd>
                            </>
                          )}
                          {data.pubchem.tpsa != null && (
                            <>
                              <dt className="text-slate-500">TPSA</dt>
                              <dd className="text-slate-200">
                                {data.pubchem.tpsa}{" "}
                                <span className="text-slate-500">Å²</span>
                              </dd>
                            </>
                          )}
                          {data.pubchem.complexity != null && (
                            <>
                              <dt className="text-slate-500">Complexity</dt>
                              <dd className="text-slate-200">
                                {data.pubchem.complexity}
                              </dd>
                            </>
                          )}
                          {data.pubchem.hBondDonorCount != null && (
                            <>
                              <dt className="text-slate-500">
                                H-bond donors
                              </dt>
                              <dd className="text-slate-200">
                                {data.pubchem.hBondDonorCount}
                              </dd>
                            </>
                          )}
                          {data.pubchem.hBondAcceptorCount != null && (
                            <>
                              <dt className="text-slate-500">
                                H-bond acceptors
                              </dt>
                              <dd className="text-slate-200">
                                {data.pubchem.hBondAcceptorCount}
                              </dd>
                            </>
                          )}
                          {data.pubchem.rotatableBondCount != null && (
                            <>
                              <dt className="text-slate-500">
                                Rotatable bonds
                              </dt>
                              <dd className="text-slate-200">
                                {data.pubchem.rotatableBondCount}
                              </dd>
                            </>
                          )}
                          {data.pubchem.exactMass != null && (
                            <>
                              <dt className="text-slate-500">Exact mass</dt>
                              <dd className="text-slate-200 font-mono text-xs">
                                {data.pubchem.exactMass}
                              </dd>
                            </>
                          )}
                          {data.pubchem.monoisotopicMass != null &&
                            data.pubchem.monoisotopicMass !==
                              data.pubchem.exactMass && (
                              <>
                                <dt className="text-slate-500">
                                  Monoisotopic mass
                                </dt>
                                <dd className="text-slate-200 font-mono text-xs">
                                  {data.pubchem.monoisotopicMass}
                                </dd>
                              </>
                            )}
                        </dl>
                      </div>
                    )}
                  </>
                ) : (
                  <p className="text-amber-300 text-sm">{data.pubchem.error}</p>
                )}
              </div>
            </Card>

            {/* RCSB — gene / protein structure & ligand SMILES */}
            <Card
              color="slate"
              title="Gene product · PDB (RCSB)"
              icon={<Dna className="w-5 h-5 text-blue-400" />}
              iconColor="blue"
            >
              <div className="space-y-4">
                <p className="text-sm text-slate-400">{data.rcsb.note}</p>

                {data.geneProduct && (
                  <div className="rounded-lg border border-teal-500/30 bg-slate-900/40 p-4 space-y-2">
                    <div className="text-xs font-medium text-slate-500 uppercase tracking-wide">
                      Protein (UniProt)
                    </div>
                    {data.geneProduct.proteinName && (
                      <p className="text-sm text-white font-medium leading-snug">
                        {data.geneProduct.proteinName}
                      </p>
                    )}
                    <dl className="grid grid-cols-1 sm:grid-cols-2 gap-x-6 gap-y-1 text-sm">
                      {data.geneProduct.geneSymbol && (
                        <>
                          <dt className="text-slate-500">Gene</dt>
                          <dd className="font-mono text-teal-300">
                            {data.geneProduct.geneSymbol}
                          </dd>
                        </>
                      )}
                      {data.geneProduct.organism && (
                        <>
                          <dt className="text-slate-500">Organism</dt>
                          <dd className="text-slate-300">
                            {data.geneProduct.organism}
                          </dd>
                        </>
                      )}
                      {data.geneProduct.sequenceLength != null && (
                        <>
                          <dt className="text-slate-500">Sequence length</dt>
                          <dd className="text-slate-200">
                            {data.geneProduct.sequenceLength}{" "}
                            <span className="text-slate-500">aa</span>
                          </dd>
                        </>
                      )}
                    </dl>
                    <a
                      href={data.geneProduct.uniprotUrl}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-flex items-center gap-2 text-sm text-teal-400 hover:underline"
                    >
                      Open {data.geneProduct.accession} on UniProt
                      <ExternalLink className="w-4 h-4" />
                    </a>
                  </div>
                )}

                {data.geneProduct?.sequence &&
                  data.geneProduct.sequence.length > 0 && (
                    <div className="rounded-lg border border-slate-600/50 bg-slate-950/40 p-4 space-y-3">
                      <div className="flex flex-wrap items-center justify-between gap-2">
                        <div className="text-xs font-medium text-slate-500 uppercase tracking-wide">
                          Sequence
                        </div>
                        <button
                          type="button"
                          onClick={async () => {
                            try {
                              await navigator.clipboard.writeText(
                                data.geneProduct!.sequence!.replace(/\s+/g, "")
                              );
                              setSequenceCopied(true);
                              window.setTimeout(() => setSequenceCopied(false), 2000);
                            } catch {
                              /* ignore */
                            }
                          }}
                          className="inline-flex items-center gap-1.5 text-xs text-teal-400 hover:text-teal-300 border border-slate-600/60 rounded-md px-2.5 py-1 bg-slate-800/60"
                        >
                          {sequenceCopied ? (
                            <>
                              <Check className="w-3.5 h-3.5" />
                              Copied
                            </>
                          ) : (
                            <>
                              <Copy className="w-3.5 h-3.5" />
                              Copy sequence
                            </>
                          )}
                        </button>
                      </div>
                      <p className="text-xs text-slate-500">
                        Sequence status:{" "}
                        <span className="text-slate-400">Complete</span>
                      </p>
                      <dl className="grid grid-cols-1 sm:grid-cols-3 gap-2 text-sm border-t border-slate-700/40 pt-3">
                        <div>
                          <dt className="text-slate-500 text-xs">Length</dt>
                          <dd className="text-slate-200 font-mono">
                            {data.geneProduct.sequenceLength ??
                              data.geneProduct.sequence.replace(/\s+/g, "")
                                .length}
                          </dd>
                        </div>
                        {data.geneProduct.sequenceMassDa != null && (
                          <div>
                            <dt className="text-slate-500 text-xs">
                              Mass (Da)
                            </dt>
                            <dd className="text-slate-200 font-mono">
                              {data.geneProduct.sequenceMassDa.toLocaleString()}
                            </dd>
                          </div>
                        )}
                        {data.geneProduct.sequenceMd5 && (
                          <div className="sm:col-span-1">
                            <dt className="text-slate-500 text-xs">
                              MD5 checksum
                            </dt>
                            <dd className="text-slate-300 font-mono text-xs break-all">
                              {data.geneProduct.sequenceMd5}
                            </dd>
                          </div>
                        )}
                      </dl>
                      <div className="rounded-md bg-slate-900/80 border border-slate-700/50 p-3 overflow-x-auto">
                        <pre className="font-mono text-[11px] sm:text-xs leading-relaxed text-slate-300 whitespace-pre">
                          {formatUniProtSequenceLines(data.geneProduct.sequence)
                            .map(
                              (row) =>
                                `${row.ruler}\n${row.aa}`
                            )
                            .join("\n\n")}
                        </pre>
                      </div>
                      <p className="text-[11px] text-slate-600">
                        Single-letter amino acid sequence from{" "}
                        <a
                          href={data.geneProduct.uniprotUrl}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-teal-500/90 hover:underline"
                        >
                          UniProt
                        </a>{" "}
                        (same formatting style as the Sequence section on{" "}
                        <a
                          href={`https://www.uniprot.org/uniprotkb/${data.geneProduct.accession}/entry#sequence`}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-teal-500/90 hover:underline"
                        >
                          the entry page
                        </a>
                        ).
                      </p>
                    </div>
                  )}

                {data.rcsb.representativeStructure && (
                  <div className="rounded-lg border border-blue-500/25 bg-slate-900/40 p-4 space-y-2">
                    <div className="text-xs font-medium text-slate-500 uppercase tracking-wide">
                      Representative PDB entry
                    </div>
                    <p className="text-sm text-white font-mono">
                      {data.rcsb.representativeStructure.pdbId}
                    </p>
                    {data.rcsb.representativeStructure.title && (
                      <p className="text-sm text-slate-300 leading-snug">
                        {data.rcsb.representativeStructure.title}
                      </p>
                    )}
                    <div className="flex flex-wrap gap-x-4 gap-y-1 text-sm text-slate-400">
                      {data.rcsb.representativeStructure.experimentalMethod && (
                        <span>
                          Method:{" "}
                          <span className="text-slate-300">
                            {
                              data.rcsb.representativeStructure
                                .experimentalMethod
                            }
                          </span>
                        </span>
                      )}
                      {data.rcsb.representativeStructure.resolutionAngstrom !=
                        null && (
                        <span>
                          Resolution:{" "}
                          <span className="text-slate-300">
                            {
                              data.rcsb.representativeStructure
                                .resolutionAngstrom
                            }{" "}
                            Å
                          </span>
                        </span>
                      )}
                    </div>
                  </div>
                )}

                {data.geneQuery && (
                  <p className="text-sm text-slate-500 bg-slate-900/50 border border-slate-700/50 rounded-lg px-3 py-2">
                    Lookup token:{" "}
                    <span className="text-blue-300 font-mono font-medium">
                      {data.geneQuery}
                    </span>
                    {data.geneLookupMode === "manual" ? (
                      <span className="text-slate-500"> (manual)</span>
                    ) : (
                      <span className="text-slate-500"> (auto)</span>
                    )}
                    {data.rcsb.uniprotAccession && (
                      <>
                        {" "}
                        · UniProt{" "}
                        <a
                          href={`https://www.uniprot.org/uniprotkb/${data.rcsb.uniprotAccession}`}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-teal-400 hover:underline font-mono"
                        >
                          {data.rcsb.uniprotAccession}
                        </a>
                      </>
                    )}
                    {data.rcsb.searchStrategy === "uniprot_accession_match" && (
                      <span className="text-slate-600">
                        {" "}
                        · matched via UniProt accession
                      </span>
                    )}
                    {data.rcsb.searchStrategy === "full_text_gene" && (
                      <span className="text-slate-600">
                        {" "}
                        · matched via RCSB full-text (broader)
                      </span>
                    )}
                  </p>
                )}

                <div className="flex flex-wrap gap-3">
                  <a
                    href={data.rcsb.searchUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-blue-500/15 text-blue-300 border border-blue-500/40 hover:bg-blue-500/25 text-sm font-medium"
                  >
                    Search RCSB.org for this gene
                    <ExternalLink className="w-4 h-4" />
                  </a>
                  {data.rcsb.structureUrl && (
                    <a
                      href={data.rcsb.structureUrl}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-slate-700/50 text-white border border-slate-600/50 hover:bg-slate-600/50 text-sm font-medium"
                    >
                      Open structure {data.rcsb.primaryPdbId} (Mol* viewer)
                      <ExternalLink className="w-4 h-4" />
                    </a>
                  )}
                  <button
                    type="button"
                    onClick={() =>
                      router.push(
                        `/dashboard/depmap/compound/embeddings?gene=${encodeURIComponent(
                          gene
                        )}&compound=${encodeURIComponent(compound)}`
                      )
                    }
                    className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-purple-600/80 text-white border border-purple-500/50 hover:bg-purple-500/80 text-sm font-medium"
                  >
                    <Search className="w-4 h-4" />
                    Find embeddings
                  </button>
                </div>

                {data.rcsb.pdbIds.length > 0 && (
                  <div>
                    <div className="text-xs font-medium text-slate-500 uppercase tracking-wide mb-2">
                      Representative PDB IDs (gene match)
                    </div>
                    <div className="flex flex-wrap gap-2">
                      {data.rcsb.pdbIds.slice(0, 8).map((id) => (
                        <a
                          key={id}
                          href={`https://www.rcsb.org/structure/${id}`}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="px-3 py-1 rounded-md bg-slate-800/80 text-teal-400 border border-slate-600/50 text-sm font-mono hover:bg-slate-700/80"
                        >
                          {id}
                        </a>
                      ))}
                    </div>
                  </div>
                )}

                {data.rcsb.ligands.length > 0 ? (
                  <div>
                    <div className="text-xs font-medium text-slate-500 uppercase tracking-wide mb-2">
                      Bound ligands (chemical components in entry{" "}
                      {data.rcsb.primaryPdbId})
                    </div>
                    <ul className="space-y-4">
                      {data.rcsb.ligands.map((lig) => (
                        <li
                          key={lig.compId}
                          className="border border-slate-700/50 rounded-lg p-4 bg-slate-900/40"
                        >
                          <div className="flex flex-wrap justify-between gap-2 mb-2">
                            <span className="text-white font-medium">
                              {lig.compId}
                            </span>
                            <a
                              href={`https://www.rcsb.org/ligand/${lig.compId}`}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="text-xs text-blue-400 hover:underline inline-flex items-center gap-1"
                            >
                              Ligand summary
                              <ExternalLink className="w-3 h-3" />
                            </a>
                          </div>
                          <p className="text-sm text-slate-400 mb-2">
                            {lig.name}
                          </p>
                          {lig.smiles ? (
                            <pre className="text-xs text-slate-200 bg-slate-950/80 border border-slate-700/60 rounded p-3 overflow-x-auto whitespace-pre-wrap break-all font-mono">
                              {lig.smiles}
                            </pre>
                          ) : (
                            <p className="text-xs text-slate-500">
                              No SMILES in RCSB descriptor for this component.
                            </p>
                          )}
                        </li>
                      ))}
                    </ul>
                  </div>
                ) : data.rcsb.primaryPdbId ? (
                  <p className="text-sm text-slate-500">
                    No bound non-polymer components listed for this entry, or
                    ligand metadata could not be loaded.
                  </p>
                ) : (
                  <p className="text-sm text-amber-300/90">
                    No PDB entries matched this gene symbol via RCSB Search API.
                    Try the RCSB search link above or verify the gene symbol.
                  </p>
                )}
              </div>
            </Card>
          </div>
        )}
      </div>
    </div>
  );
}
