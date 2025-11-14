import { useState } from "react";
import { X, Database, Upload, FileText, Calendar } from "lucide-react";

interface Dataset {
  updatedAt(updatedAt: any): import("react").ReactNode;
  id: string;
  name: string;
  description: string;
  uploadedAt: string;
  rowCount: number;
  columnCount: number;
}

interface DatasetSelectionModalProps {
  onClose: () => void;
  onSelectDataset: (datasetId: string) => void;
  onUploadNew: () => void;
}

export function DatasetSelectionModal({
  onClose,
  onSelectDataset,
  onUploadNew,
}: DatasetSelectionModalProps) {
  const [datasets] = useState<Dataset[]>(() => {
    const stored = sessionStorage.getItem("datasets");
    console.log("stored");
    console.log(stored ? JSON.parse(stored)[0]?.uploadedAt : "No stored datasets");
    return stored
      ? JSON.parse(stored)
      : [];
  });

  const formatDate = (date: string) => {
    console.log("formatDate input:", date);
    return new Date(date).toLocaleDateString("en-US", {
      year: "numeric",
      month: "short",
      day: "numeric",
    });
  };

  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-xl shadow-2xl max-w-3xl w-full border border-slate-700/50">
        <div className="border-b border-slate-700/50 p-6 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-gradient-to-br from-teal-500/20 to-blue-500/20 rounded-lg">
              <Database className="w-6 h-6 text-teal-400" />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-white">Select Dataset</h2>
              <p className="text-sm text-slate-400">
                Choose an existing dataset or upload new data
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-slate-700/50 rounded-lg transition-colors"
          >
            <X className="w-6 h-6 text-slate-400" />
          </button>
        </div>

        <div className="p-6">
          <button
            onClick={onUploadNew}
            className="w-full mb-6 p-6 bg-gradient-to-r from-teal-600/20 to-blue-600/20 hover:from-teal-600/30 hover:to-blue-600/30 border-2 border-dashed border-teal-500/50 hover:border-teal-500 rounded-xl transition-all group"
          >
            <div className="flex items-center justify-center gap-3">
              <Upload className="w-6 h-6 text-teal-400 group-hover:scale-110 transition-transform" />
              <div className="text-left">
                <div className="text-lg font-semibold text-white">
                  Upload New Dataset
                </div>
                <div className="text-sm text-slate-400">
                  Upload CSV, TSV, or Excel files
                </div>
              </div>
            </div>
          </button>

          <div className="mb-4">
            <h3 className="text-sm font-medium text-slate-400 uppercase tracking-wider">
              Your Datasets
            </h3>
          </div>

          <div className="space-y-3 max-h-96 overflow-y-auto">
            {datasets.map((dataset) => (
              <button
                key={dataset.id}
                onClick={() => onSelectDataset(dataset.id)}
                className="w-full p-4 bg-slate-700/30 hover:bg-slate-700/50 border border-slate-600/50 hover:border-teal-500/50 rounded-lg transition-all text-left group"
              >
                <div className="flex items-start justify-between mb-2">
                  <div className="flex items-center gap-3">
                    <FileText className="w-5 h-5 text-teal-400 flex-shrink-0" />
                    <div>
                      <h4 className="text-white font-semibold group-hover:text-teal-400 transition-colors">
                        {dataset.name}
                      </h4>
                      <p className="text-sm text-slate-400 mt-1">
                        {dataset.description}
                      </p>
                    </div>
                  </div>
                </div>
                <div className="flex items-center gap-4 text-xs text-slate-500 mt-3">
                  <div className="flex items-center gap-1">
                    <Calendar className="w-3.5 h-3.5" />
                    {/* @ts-ignore */}
                    <span>{formatDate(dataset.updatedAt)}</span>
                  </div>
                  <span>•</span>
                  <span>{dataset.rowCount.toLocaleString()} samples</span>
                  <span>•</span>
                  <span>{dataset.columnCount.toLocaleString()} genes</span>
                </div>
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
