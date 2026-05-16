"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { Card } from "@repo/ui/card";
import { AFFINITY_BACKEND_URL } from "@repo/config";
import { ArrowLeft, Loader, AlertCircle, Upload, Download } from "lucide-react";

type PredictResponse = {
  predicted_affinity: number;
  row_count: number;
  required_columns: string[];
};

export default function AffinityPage() {
  const router = useRouter();
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);

  async function handlePredict() {
    if (!file) {
      setError("Please upload a CSV file first.");
      return;
    }
    const token = sessionStorage.getItem("authToken");
    if (!token) {
      setError("Missing authentication token. Please login again.");
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const form = new FormData();
      form.append("file", file);

      const res = await fetch(`${AFFINITY_BACKEND_URL}/affinity/predict`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${token}`,
        },
        body: form,
      });

      if (!res.ok) {
        const txt = await res.text().catch(() => "");
        throw new Error(`Prediction failed: HTTP ${res.status} ${txt}`);
      }

      const json = (await res.json()) as PredictResponse;
      setResult(json);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed to predict affinity.");
    } finally {
      setLoading(false);
    }
  }

  async function handleDownloadSample() {
    const token = sessionStorage.getItem("authToken");
    if (!token) {
      setError("Missing authentication token. Please login again.");
      return;
    }

    try {
      const res = await fetch(`${AFFINITY_BACKEND_URL}/affinity/sample-csv`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (!res.ok) {
        const txt = await res.text().catch(() => "");
        throw new Error(`Sample download failed: HTTP ${res.status} ${txt}`);
      }

      const blob = await res.blob();
      const objectUrl = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = objectUrl;
      a.download = "affinity_input_sample.csv";
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(objectUrl);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed to download sample CSV.");
    }
  }

  return (
    <div className="h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 overflow-y-auto">
      <div className="max-w-5xl mx-auto p-8">
        <div className="mb-6">
          <button
            type="button"
            onClick={() => router.push("/dashboard")}
            className="inline-flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-lg bg-slate-800/50 text-slate-300 border border-slate-700/50 hover:bg-slate-700/50 transition-colors"
          >
            <ArrowLeft className="w-4 h-4" />
            Back to Dashboard
          </button>
        </div>

        <Card color="slate" className="mb-6">
          <div className="space-y-4">
            <h1 className="text-2xl font-bold text-white">Find Affinity</h1>
            <p className="text-slate-400">
              Upload embeddings CSV, run affinity prediction from the model in
              <span className="font-mono text-slate-300"> apps/affinity</span>,
              and download predictions.
            </p>

            <div className="rounded-lg border border-slate-700/50 bg-slate-900/40 p-4 space-y-2">
              <p className="text-sm text-slate-300 font-medium">CSV format</p>
              <p className="text-xs text-slate-500">
                Required id columns: <span className="font-mono">drug_id</span> and
                <span className="font-mono"> gene_id</span> (or
                <span className="font-mono"> protein_id</span>). Numeric embedding
                columns should use prefixes:
                <span className="font-mono"> drug_mol2vec_*</span>,
                <span className="font-mono"> drug_gin_*</span>,
                <span className="font-mono"> drug_unimol_*</span>,
                <span className="font-mono"> gene_protvec_*</span>,
                <span className="font-mono"> gene_protbert_*</span>,
                <span className="font-mono"> gene_esm_*</span>.
              </p>
              <button
                type="button"
                onClick={handleDownloadSample}
                className="inline-flex items-center gap-2 px-3 py-2 rounded-lg bg-slate-700/50 text-slate-200 border border-slate-600/50 hover:bg-slate-600/50 text-sm"
              >
                <Download className="w-4 h-4" />
                Download sample CSV
              </button>
            </div>

            <div className="flex flex-col sm:flex-row gap-3 items-start sm:items-center">
              <label className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-slate-800/60 border border-slate-700/50 text-slate-200 cursor-pointer hover:bg-slate-700/60">
                <Upload className="w-4 h-4" />
                <span>{file ? file.name : "Choose CSV file"}</span>
                <input
                  type="file"
                  accept=".csv,text/csv"
                  className="hidden"
                  onChange={(e) => setFile(e.target.files?.[0] ?? null)}
                />
              </label>

              <button
                type="button"
                disabled={loading || !file}
                onClick={handlePredict}
                className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-teal-600/80 text-white border border-teal-500/50 hover:bg-teal-500/80 disabled:opacity-50 text-sm font-medium"
              >
                {loading ? (
                  <>
                    <Loader className="w-4 h-4 animate-spin" />
                    Predicting...
                  </>
                ) : (
                  <>Run Affinity Prediction</>
                )}
              </button>
            </div>
          </div>
        </Card>

        {error && (
          <Card color="slate" className="mb-6">
            <div className="flex items-start gap-3 bg-red-500/10 border border-red-500/30 rounded-lg p-4">
              <AlertCircle className="w-5 h-5 text-red-400 shrink-0 mt-0.5" />
              <p className="text-red-300 text-sm whitespace-pre-wrap">{error}</p>
            </div>
          </Card>
        )}

        {result && (
          <Card color="slate">
            <div className="space-y-4">
              <h2 className="text-lg font-semibold text-white">Prediction Complete</h2>

              <p className="text-sm text-slate-400">
                Rows processed: <span className="text-slate-200">{result.row_count}</span>
              </p>

              <div className="rounded-lg border border-emerald-500/30 bg-emerald-500/10 p-6 text-center">
                <p className="text-xs uppercase tracking-wide text-emerald-300 mb-2">
                  Final predicted affinity
                </p>
                <div className="text-4xl font-bold text-white font-mono">
                  {result.predicted_affinity.toFixed(6)}
                </div>
              </div>
            </div>
          </Card>
        )}
      </div>
    </div>
  );
}
