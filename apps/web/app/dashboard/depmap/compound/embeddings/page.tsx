"use client";

import { useEffect, useState } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import { Card } from "@repo/ui/card";
import { Loader, ExternalLink, AlertCircle, Search } from "lucide-react";

function objToCsv(rows: Array<Record<string, any>>): string {
  if (!rows || rows.length === 0) return "";
  const keys = Array.from(
    rows.reduce((acc, r) => {
      Object.keys(r).forEach((k) => acc.add(k));
      return acc;
    }, new Set<string>())
  );
  const header = keys.join(",");
  const lines = rows.map((r) =>
    keys
      .map((k) => {
        const v = r[k];
        if (v === null || v === undefined) return "";
        if (Array.isArray(v)) return `"${v.join(";")}"`;
        return `"${String(v).replace(/"/g, '""')}"`;
      })
      .join(",")
  );
  return [header, ...lines].join("\n");
}

function downloadFile(filename: string, content: string, mime = "text/csv") {
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

export default function EmbeddingsPage() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const gene = searchParams.get("gene") || "";
  const compound = searchParams.get("compound") || "";

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [molecular, setMolecular] = useState<any | null>(null);
  const [result, setResult] = useState<any | null>(null);

  useEffect(() => {
    if (!gene || !compound) return;
    setLoading(true);
    setError(null);
    fetch(`/api/depmap/molecular?gene=${encodeURIComponent(gene)}&compound=${encodeURIComponent(compound)}`)
      .then((r) => r.json())
      .then((j) => setMolecular(j))
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false));
  }, [gene, compound]);

  async function runEmbeddings() {
    if (!molecular) return;
    const token = typeof window !== "undefined" ? sessionStorage.getItem("authToken") : null;
    if (!token) {
      setError("Missing authentication token. Please sign in.");
      return;
    }

    const canonical = molecular.pubchem?.canonicalSmiles ?? molecular.pubchem?.isomericSmiles ?? molecular.rcsb?.primaryLigand?.smiles ?? "";
    const geneSeq = molecular.geneProduct?.sequence ?? "";
    const drugId = molecular.compoundQuery ?? molecular.compound;
    const geneId = molecular.geneQuery ?? molecular.gene;

    if (!canonical) {
      setError("No canonical SMILES available for the compound.");
      return;
    }
    if (!geneSeq) {
      setError("No gene sequence available for this gene.");
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);
    try {
      // enqueue async job so the worker will produce artifacts (including zip)
      const enqueueRes = await fetch("http://localhost:8002/embeddings/async", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          drug_id: drugId,
          canonical_smiles: canonical,
          gene_id: geneId,
          gene_sequence: geneSeq,
          include_vectors: true,
          include_combined_csv: true,
          create_zip: true,
        }),
      });

      if (!enqueueRes.ok) {
        const body = await enqueueRes.text().catch(() => "");
        throw new Error(`Enqueue failed: HTTP ${enqueueRes.status} ${body}`);
      }
      const enqueueJson = await enqueueRes.json();
      const jobId = enqueueJson.job_id;
      if (!jobId) throw new Error("No job_id returned from enqueue.");

      // poll status
      let finished = false;
      const start = Date.now();
      const timeoutMs = 45 * 60 * 1000; // 45 minutes (job timeout on server)
      while (!finished) {
        await new Promise((r) => setTimeout(r, 1500));
        const statusRes = await fetch(`http://localhost:8002/embeddings/${encodeURIComponent(jobId)}/status`, {
          headers: { Authorization: `Bearer ${token}` },
        });
        if (!statusRes.ok) {
          const body = await statusRes.text().catch(() => "");
          throw new Error(`Status check failed: HTTP ${statusRes.status} ${body}`);
        }
        const statusJson = await statusRes.json();
        const status = statusJson.status;
        if (status === "finished") {
          finished = true;
          // store result for metadata display
          setResult(statusJson.result ?? null);
          // download zip artifact
          const dlRes = await fetch(`http://localhost:8002/embeddings/${encodeURIComponent(jobId)}/download?format=zip`, {
            headers: { Authorization: `Bearer ${token}` },
          });
          if (!dlRes.ok) {
            const body = await dlRes.text().catch(() => "");
            throw new Error(`Download failed: HTTP ${dlRes.status} ${body}`);
          }
          const blob = await dlRes.blob();
          // build filename from drug and gene
          const sanitize = (s: string) =>
            String(s || "").replace(/\s+/g, "_").replace(/[^A-Za-z0-9_\-.]/g, "");
          const drugName = sanitize(molecular.compoundQuery ?? molecular.compound ?? "drug");
          const geneName = sanitize(molecular.geneQuery ?? molecular.gene ?? "gene");
          const filename = `${drugName}_${geneName}_embeddings.zip`;
          const url = URL.createObjectURL(blob);
          const a = document.createElement("a");
          a.href = url;
          a.download = filename;
          document.body.appendChild(a);
          a.click();
          a.remove();
          URL.revokeObjectURL(url);
        } else if (status === "failed") {
          finished = true;
          const err = statusJson.result?.error ?? "Job failed";
          throw new Error(String(err));
        } else {
          // still queued or started; check timeout
          if (Date.now() - start > timeoutMs) {
            throw new Error("Timeout waiting for embedding job to finish.");
          }
        }
      }
    } catch (e: any) {
      setError(e?.message || String(e));
    } finally {
      setLoading(false);
    }
  }

  function downloadCSVs() {
    if (!result) return;
    const metadata = result.metadata ?? {};
    const vectors = result.vectors ?? {};

    // Metadata CSV
    const metadataCsv = objToCsv([metadata]);
    downloadFile("metadata.csv", metadataCsv);

    // Drug CSV
    const drugRow: Record<string, any> = {
      drug_id: metadata.drug_id,
      canonical_smiles: metadata.canonical_smiles,
    };
    for (const k of Object.keys(vectors)) {
      if (k.startsWith("drug_")) {
        // add entries like drug_unimol_0... as columns
        const arr = vectors[k];
        for (let i = 0; i < arr.length; i++) {
          drugRow[`${k}_${i}`] = arr[i];
        }
      }
    }
    const drugCsv = objToCsv([drugRow]);
    downloadFile("drug_embeddings.csv", drugCsv);

    // Gene CSV
    const geneRow: Record<string, any> = {
      gene_id: metadata.gene_id,
      gene_sequence: metadata.gene_sequence,
    };
    for (const k of Object.keys(vectors)) {
      if (k.startsWith("gene_")) {
        const arr = vectors[k];
        for (let i = 0; i < arr.length; i++) {
          geneRow[`${k}_${i}`] = arr[i];
        }
      }
    }
    const geneCsv = objToCsv([geneRow]);
    downloadFile("gene_embeddings.csv", geneCsv);

    // Combined CSV
    const combined: Record<string, any> = { ...metadata };
    Object.assign(combined, drugRow, geneRow);
    const combinedCsv = objToCsv([combined]);
    downloadFile("combined_embeddings.csv", combinedCsv);
  }

  return (
    <div className="h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 overflow-y-auto">
      <div className="max-w-4xl mx-auto p-8">
        <div className="mb-6">
          <button
            type="button"
            onClick={() => router.back()}
            className="inline-flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-lg bg-slate-800/50 text-slate-300 border border-slate-700/50 hover:bg-slate-700/50 transition-colors"
          >
            Back
          </button>
        </div>

        <Card color="slate">
          <div className="space-y-4">
            <h1 className="text-2xl font-bold text-white">Embedding bundle</h1>
            <p className="text-slate-400">Generate drug & gene embeddings for the selected pair.</p>

            {loading && (
              <div className="text-center py-6">
                <Loader className="w-10 h-10 text-teal-400 mx-auto animate-spin" />
                <p className="text-slate-400 mt-2">Working…</p>
              </div>
            )}

            {error && !loading && (
              <div className="rounded-lg border border-red-500/30 bg-red-500/10 p-4">
                <div className="flex items-start gap-3">
                  <AlertCircle className="w-5 h-5 text-red-400" />
                  <div className="text-red-300">{error}</div>
                </div>
              </div>
            )}

            {!loading && molecular && (
              <div className="space-y-3">
                <div className="text-sm text-slate-500">
                  Compound: <span className="text-teal-300 font-mono">{molecular.compoundQuery ?? molecular.compound}</span>
                </div>
                <div className="text-sm text-slate-500">
                  Gene: <span className="text-blue-300 font-mono">{molecular.geneQuery ?? molecular.gene}</span>
                </div>
                <div className="flex gap-2">
                  <button
                    className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-purple-600/80 text-white border border-purple-500/50 hover:bg-purple-500/80 text-sm font-medium"
                    onClick={runEmbeddings}
                  >
                    <Search className="w-4 h-4" />
                    Generate & Download ZIP
                  </button>
                </div>

                {result && (
                  <div className="rounded-lg border border-slate-700/50 bg-slate-900/40 p-4">
                    <div className="text-sm text-slate-300">Result metadata</div>
                    <pre className="text-xs text-slate-200 mt-2 font-mono whitespace-pre-wrap break-all">{JSON.stringify(result.metadata, null, 2)}</pre>
                  </div>
                )}
              </div>
            )}

            {!loading && !molecular && (
              <p className="text-slate-400">Loading molecular data…</p>
            )}
          </div>
        </Card>
      </div>
    </div>
  );
}
