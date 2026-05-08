"use client";

import { useState, useEffect, useRef } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import axios from "axios";
import { Card } from "@repo/ui/card";
import {
  Loader,
  Download,
  ArrowLeft,
  CheckCircle,
  XCircle,
  Clock,
} from "lucide-react";
import { DEPMAP_BACKEND_URL } from "@repo/config";

interface AssociationResult {
  "Gene/Compound": string;
  Dataset: string;
  Correlation: number;
  other_entity_type: string;
}

export default function DepMapPage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const geneSymbol = searchParams.get("gene") || "";
  const experimentId = searchParams.get("experimentId") || "default";

  const [loading, setLoading] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const [jobStatus, setJobStatus] = useState<string>("");
  const [associations, setAssociations] = useState<AssociationResult[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [polling, setPolling] = useState(false);
  const [forceExport, setForceExport] = useState(false);
  const [existingFile, setExistingFile] = useState<string | null>(null);
  
  // Use ref to prevent duplicate API calls (React Strict Mode in dev runs effects twice)
  const hasStartedRef = useRef(false);
  const analysisKeyRef = useRef<string>("");

  useEffect(() => {
    // Create a unique key for this analysis (gene + experiment)
    const currentKey = `${geneSymbol}-${experimentId}`;
    
    // Reset when gene or experiment changes
    if (analysisKeyRef.current !== currentKey) {
      hasStartedRef.current = false;
      setJobId(null);
      setJobStatus("");
      setAssociations([]);
      setError(null);
      setExistingFile(null);
      setForceExport(false);
      analysisKeyRef.current = currentKey;
    }
    
    // Only start if we have a gene symbol, no job ID yet, and haven't started this specific analysis
    // AND force export is not enabled (if force is enabled, user will manually trigger)
    if (geneSymbol && !jobId && !forceExport && (!hasStartedRef.current || analysisKeyRef.current !== currentKey)) {
      hasStartedRef.current = true;
      analysisKeyRef.current = currentKey;
      startAnalysis();
    }
  }, [geneSymbol, experimentId]);

  useEffect(() => {
    if (jobId && (jobStatus === "queued" || jobStatus === "started")) {
      setPolling(true);
      const interval = setInterval(async () => {
        await checkJobStatus();
      }, 3000); // Poll every 3 seconds

      return () => {
        clearInterval(interval);
        setPolling(false);
      };
    }
  }, [jobId, jobStatus]);

  const startAnalysis = async () => {
    if (!geneSymbol) {
      setError("No gene symbol provided");
      return;
    }

    // Prevent duplicate calls
    if (loading || jobId) {
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const token = sessionStorage.getItem("authToken");
      if (!token) {
        setError("Not authenticated. Please log in.");
        router.push("/login");
        return;
      }

      // Trim whitespace from gene symbol and normalize to uppercase for consistency
      const cleanedGene = geneSymbol.trim().toUpperCase();
      
      const response = await axios.post(
        `${DEPMAP_BACKEND_URL}/associations`,
        { 
          genes: [cleanedGene],
          experiment_id: experimentId,
          force: forceExport
        },
        {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        }
      );

      // Check if results already exist (cached)
      if (response.data.status === "finished" && response.data.existing_files) {
        // The backend returns existing_files with normalized (uppercase) keys
        const filePath = response.data.existing_files[cleanedGene];
        setExistingFile(filePath || null);
        setJobStatus("finished");
        setJobId("cached");
        // Load existing results from the cached file
        if (filePath) {
          try {
            await loadCachedResults(cleanedGene);
          } catch (err) {
            console.error("Failed to load cached results:", err);
            setError("Failed to load cached results. You can download the CSV or regenerate.");
          }
        }
      } else {
        setJobId(response.data.job_id);
        setJobStatus(response.data.status || "queued");
      }
    } catch (err: any) {
      console.error("Error starting DepMap analysis:", err);
      setError(
        err.response?.data?.detail || "Failed to start analysis. Please try again."
      );
      // Reset ref on error so user can retry
      hasStartedRef.current = false;
    } finally {
      setLoading(false);
    }
  };

  const checkJobStatus = async () => {
    if (!jobId) return;

    try {
      const token = sessionStorage.getItem("authToken");
      const response = await axios.get(
        `${DEPMAP_BACKEND_URL}/associations/${jobId}/status`,
        {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        }
      );

      const status = response.data.status;
      setJobStatus(status);

      if (status === "finished") {
        setPolling(false);
        // Load results from CSV (we'll parse it or fetch as JSON if available)
        await loadResults();
      } else if (status === "failed") {
        setPolling(false);
        setError("Analysis failed. Please try again.");
      }
    } catch (err: any) {
      console.error("Error checking job status:", err);
    }
  };

  const loadCachedResults = async (geneName: string) => {
    if (!experimentId || experimentId === "default") {
      return;
    }
    
    try {
      const token = sessionStorage.getItem("authToken");
      if (!token) {
        setError("Not authenticated. Please log in.");
        return;
      }
      
      // Gene name is already normalized to uppercase
      const url = `${DEPMAP_BACKEND_URL}/associations/experiment/${experimentId}/gene/${encodeURIComponent(geneName)}/download`;
      
      const response = await axios.get(url, {
        headers: {
          Authorization: `Bearer ${token}`,
        },
        responseType: "blob",
      });
      
      // Parse CSV blob - handle quoted fields properly
      const text = await response.data.text();
      const lines = text.split("\n").filter(line => line.trim());
      
      if (lines.length < 2) {
        setError("CSV file is empty or invalid");
        return;
      }

      const data: AssociationResult[] = [];
      
      // Simple CSV parser that handles quoted fields
      const parseCSVLine = (line: string): string[] => {
        const result: string[] = [];
        let current = "";
        let inQuotes = false;
        
        for (let i = 0; i < line.length; i++) {
          const char = line[i];
          if (char === '"') {
            inQuotes = !inQuotes;
          } else if (char === ',' && !inQuotes) {
            result.push(current.trim());
            current = "";
          } else {
            current += char;
          }
        }
        result.push(current.trim()); // Add last field
        return result;
      };

      // Skip header line
      for (let i = 1; i < lines.length; i++) {
        const values = parseCSVLine(lines[i]);
        if (values.length >= 4) {
          data.push({
            "Gene/Compound": values[0] || "",
            Dataset: values[1] || "",
            Correlation: parseFloat(values[2] || "0"),
            other_entity_type: values[3] || "",
          });
        }
      }

      if (data.length === 0) {
        setError("CSV file appears to be empty or in an unexpected format.");
      } else {
        setAssociations(data);
        setError(null); // Clear any previous errors
      }
    } catch (err: any) {
      console.error("Error loading cached results:", err);
      if (err.response?.status === 401) {
        setError("Authentication failed. Please log in again.");
      } else if (err.response?.status === 404) {
        setError(`File not found: ${err.response?.data?.detail || "The cached file may have been deleted."}`);
      } else {
        setError(`Failed to load cached results: ${err.response?.data?.detail || err.message || "Unknown error"}. You can download the CSV or regenerate.`);
      }
    }
  };

  const loadResults = async () => {
    if (!jobId || jobId === "cached") return;

    try {
      const token = sessionStorage.getItem("authToken");
      const response = await axios.get(
        `${DEPMAP_BACKEND_URL}/associations/${jobId}/download`,
        {
          headers: {
            Authorization: `Bearer ${token}`,
          },
          responseType: "blob",
        }
      );

      // Parse CSV blob - handle quoted fields properly
      const text = await response.data.text();
      const lines = text.split("\n").filter(line => line.trim());
      if (lines.length < 2) {
        setError("CSV file is empty or invalid");
        return;
      }

      const data: AssociationResult[] = [];
      
      // Simple CSV parser that handles quoted fields
      const parseCSVLine = (line: string): string[] => {
        const result: string[] = [];
        let current = "";
        let inQuotes = false;
        
        for (let i = 0; i < line.length; i++) {
          const char = line[i];
          if (char === '"') {
            inQuotes = !inQuotes;
          } else if (char === ',' && !inQuotes) {
            result.push(current.trim());
            current = "";
          } else {
            current += char;
          }
        }
        result.push(current.trim()); // Add last field
        return result;
      };

      // Skip header line
      for (let i = 1; i < lines.length; i++) {
        const values = parseCSVLine(lines[i]);
        if (values.length >= 4) {
          data.push({
            "Gene/Compound": values[0] || "",
            Dataset: values[1] || "",
            Correlation: parseFloat(values[2] || "0"),
            other_entity_type: values[3] || "",
          });
        }
      }

      setAssociations(data);
    } catch (err: any) {
      console.error("Error loading results:", err);
      setError("Failed to load results. Please try downloading the CSV.");
    }
  };

  const handleDownload = async () => {
    try {
      const token = sessionStorage.getItem("authToken");
      let response;
      
      // Use different endpoint for cached vs new results
      if (jobId === "cached" && experimentId !== "default") {
        // Gene symbol is already normalized to uppercase
        response = await axios.get(
          `${DEPMAP_BACKEND_URL}/associations/experiment/${experimentId}/gene/${encodeURIComponent(geneSymbol.trim().toUpperCase())}/download`,
          {
            headers: {
              Authorization: `Bearer ${token}`,
            },
            responseType: "blob",
          }
        );
      } else if (jobId && jobId !== "cached") {
        response = await axios.get(
          `${DEPMAP_BACKEND_URL}/associations/${jobId}/download`,
          {
            headers: {
              Authorization: `Bearer ${token}`,
            },
            responseType: "blob",
          }
        );
      } else {
        setError("No results available to download");
        return;
      }

      const blob = new Blob([response.data], { type: "text/csv" });
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `depmap_${geneSymbol.trim()}_associations.csv`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
    } catch (err: any) {
      console.error("Error downloading CSV:", err);
      setError("Failed to download CSV. Please try again.");
    }
  };

  const getStatusIcon = () => {
    switch (jobStatus) {
      case "finished":
        return <CheckCircle className="w-5 h-5 text-emerald-400" />;
      case "failed":
        return <XCircle className="w-5 h-5 text-red-400" />;
      case "queued":
      case "started":
        return <Loader className="w-5 h-5 text-blue-400 animate-spin" />;
      default:
        return <Clock className="w-5 h-5 text-slate-400" />;
    }
  };

  const getStatusText = () => {
    switch (jobStatus) {
      case "finished":
        return "Analysis Complete";
      case "failed":
        return "Analysis Failed";
      case "queued":
        return "Queued";
      case "started":
        return "Processing...";
      default:
        return "Unknown";
    }
  };

  return (
    <div className="h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 overflow-y-auto">
      <div className="max-w-7xl mx-auto p-8">
        <div className="mb-6">
          <button
            onClick={() => router.back()}
            className="inline-flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-lg bg-slate-800/50 text-slate-300 border border-slate-700/50 hover:bg-slate-700/50 transition-colors"
          >
            <ArrowLeft className="w-4 h-4" />
            Back
          </button>
        </div>

        <Card color="slate" className="mb-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-white mb-2">
                DepMap Drug Associations
              </h1>
              <p className="text-slate-400">
                Gene: <span className="text-white font-medium">{geneSymbol}</span>
              </p>
            </div>
            <div className="flex items-center gap-3">
              {getStatusIcon()}
              <span className="text-sm font-medium text-slate-300">
                {getStatusText()}
              </span>
            </div>
          </div>
          
          {/* Always show force regenerate option when analysis is finished or when there's an existing file */}
          {(jobStatus === "finished" || existingFile) && (
            <div className="mt-4 p-3 bg-slate-800/50 rounded-lg border border-slate-700/50">
              {existingFile && !forceExport && (
                <div className="mb-3 p-2 bg-blue-500/10 border border-blue-500/30 rounded">
                  <p className="text-sm text-blue-300 mb-2">
                    Results already exist for this gene. 
                  </p>
                </div>
              )}
              <div className="flex items-center gap-3">
                <label className="flex items-center gap-2 text-sm text-slate-300 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={forceExport}
                    onChange={(e) => {
                      setForceExport(e.target.checked);
                      if (e.target.checked) {
                        // If enabling force, reset and allow manual start
                        hasStartedRef.current = false;
                        setJobId(null);
                        setJobStatus("");
                        setExistingFile(null);
                        setAssociations([]);
                      }
                    }}
                    className="w-4 h-4 rounded border-slate-600 bg-slate-700 text-purple-500 focus:ring-purple-500"
                  />
                  <span>Force regenerate (ignore existing results)</span>
                </label>
                {forceExport && (
                  <button
                    onClick={() => {
                      hasStartedRef.current = false;
                      startAnalysis();
                    }}
                    className="px-4 py-2 text-sm font-medium rounded-lg bg-purple-500/20 text-purple-300 border border-purple-500/40 hover:bg-purple-500/30 transition-colors"
                  >
                    Start Analysis
                  </button>
                )}
              </div>
            </div>
          )}
          
          {/* Show force option before starting if no job yet */}
          {!jobId && !loading && jobStatus !== "finished" && !existingFile && (
            <div className="mt-4 p-3 bg-slate-800/50 rounded-lg border border-slate-700/50">
              <label className="flex items-center gap-2 text-sm text-slate-300 cursor-pointer">
                <input
                  type="checkbox"
                  checked={forceExport}
                  onChange={(e) => {
                    setForceExport(e.target.checked);
                    if (e.target.checked) {
                      // If enabling force, reset and allow manual start
                      hasStartedRef.current = false;
                      setJobId(null);
                      setJobStatus("");
                    }
                  }}
                  className="w-4 h-4 rounded border-slate-600 bg-slate-700 text-purple-500 focus:ring-purple-500"
                />
                <span>Force regenerate (ignore existing results)</span>
              </label>
              {forceExport && (
                <button
                  onClick={() => {
                    hasStartedRef.current = false;
                    startAnalysis();
                  }}
                  className="mt-3 px-4 py-2 text-sm font-medium rounded-lg bg-purple-500/20 text-purple-300 border border-purple-500/40 hover:bg-purple-500/30 transition-colors"
                >
                  Start Analysis
                </button>
              )}
            </div>
          )}
        </Card>

        {error && (
          <Card color="slate" className="mb-6">
            <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-4">
              <p className="text-red-400">{error}</p>
            </div>
          </Card>
        )}

        {jobStatus === "finished" && (associations.length > 0 || existingFile) && (
          <>
            <Card color="slate" className="mb-6">
              <div className="flex items-center justify-between mb-4">
                <div>
                  <h2 className="text-xl font-bold text-white">
                    Drug Associations{" "}
                    {associations.length > 0
                      ? `(${associations.length} found)`
                      : "(loading...)"}
                  </h2>
                  <p className="text-sm text-slate-500 mt-1">
                    Click a row to open compound details (PubChem SMILES · RCSB
                    structures).
                  </p>
                </div>
                <button
                  onClick={handleDownload}
                  className="inline-flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-lg bg-emerald-500/10 text-emerald-300 border border-emerald-500/40 hover:bg-emerald-500/20 transition-colors"
                >
                  <Download className="w-4 h-4" />
                  Download CSV
                </button>
              </div>
            </Card>

            {associations.length > 0 ? (
              <Card color="slate">
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead>
                      <tr className="border-b border-slate-700/50">
                        <th className="text-left py-3 px-4 text-sm font-medium text-slate-300">
                          Compound
                        </th>
                        <th className="text-left py-3 px-4 text-sm font-medium text-slate-300">
                          Dataset
                        </th>
                        <th className="text-right py-3 px-4 text-sm font-medium text-slate-300">
                          Correlation
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      {associations.map((assoc, idx) => (
                        <tr
                          key={idx}
                          role="button"
                          tabIndex={0}
                          onClick={() => {
                            const c = assoc["Gene/Compound"];
                            const q = new URLSearchParams();
                            q.set("gene", geneSymbol.trim().toUpperCase());
                            q.set("compound", c);
                            if (experimentId) q.set("experimentId", experimentId);
                            q.set("dataset", assoc.Dataset);
                            q.set("correlation", String(assoc.Correlation));
                            router.push(
                              `/dashboard/depmap/compound?${q.toString()}`
                            );
                          }}
                          onKeyDown={(e) => {
                            if (e.key === "Enter" || e.key === " ") {
                              e.preventDefault();
                              (e.currentTarget as HTMLTableRowElement).click();
                            }
                          }}
                          className="border-b border-slate-800/50 hover:bg-slate-800/30 transition-colors cursor-pointer"
                        >
                          <td className="py-3 px-4 text-white font-medium">
                            {assoc["Gene/Compound"]}
                          </td>
                          <td className="py-3 px-4 text-slate-400 text-sm">
                            {assoc.Dataset}
                          </td>
                          <td className="py-3 px-4 text-right">
                            <span
                              className={`font-medium ${
                                assoc.Correlation > 0
                                  ? "text-emerald-400"
                                  : "text-red-400"
                              }`}
                            >
                              {assoc.Correlation > 0 ? "+" : ""}
                              {assoc.Correlation.toFixed(3)}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </Card>
            ) : existingFile ? (
              <Card color="slate" className="text-center py-8">
                <p className="text-slate-400 mb-4">
                  Results file exists. Click "Download CSV" to view, or use "Force regenerate" to create a new one.
                </p>
              </Card>
            ) : null}
          </>
        )}

        {jobStatus === "finished" && associations.length === 0 && !existingFile && jobId !== "cached" && (
          <Card color="slate" className="text-center py-12">
            <p className="text-slate-400">
              Results are being loaded...
            </p>
          </Card>
        )}
        
        {jobStatus === "finished" && jobId === "cached" && associations.length === 0 && existingFile && (
          <Card color="slate" className="text-center py-12">
            <p className="text-slate-400 mb-4">
              Loading cached results...
            </p>
            {error && (
              <p className="text-red-400 text-sm">{error}</p>
            )}
          </Card>
        )}

        {(jobStatus === "queued" || jobStatus === "started") && (
          <Card color="slate" className="text-center py-12">
            <Loader className="w-12 h-12 text-blue-400 mx-auto mb-4 animate-spin" />
            <p className="text-slate-400">
              Analyzing drug associations for {geneSymbol}...
            </p>
            <p className="text-sm text-slate-500 mt-2">
              This may take a few minutes. Please wait...
            </p>
          </Card>
        )}
      </div>
    </div>
  );
}

