import { useState } from "react";
import { X, Upload, FileText, CheckCircle2 } from "lucide-react";

interface DatasetUploadModalProps {
  onClose: () => void;
  onUploadComplete: (datasetId: string) => void;
}

export function DatasetUploadModal({
  onClose,
  onUploadComplete,
}: DatasetUploadModalProps) {
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setUploading(true);

    try {
      const newDatasetId = `dataset_${Date.now()}`;
      const newDataset = {
        id: newDatasetId,
        name,
        description,
        uploadedAt: new Date().toISOString(),
        rowCount: Math.floor(Math.random() * 200) + 50,
        columnCount: Math.floor(Math.random() * 10000) + 10000,
        fileName: file?.name || "unknown.csv",
      };

      const stored = sessionStorage.getItem("datasets");
      const datasets = stored ? JSON.parse(stored) : [];
      datasets.unshift(newDataset);
      sessionStorage.setItem("datasets", JSON.stringify(datasets));

      onUploadComplete(newDatasetId);
    } catch (error) {
      console.error("Upload error:", error);
      alert("Failed to upload dataset. Please try again.");
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-xl shadow-2xl max-w-2xl w-full border border-slate-700/50">
        <div className="border-b border-slate-700/50 p-6 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-gradient-to-br from-teal-500/20 to-blue-500/20 rounded-lg">
              <Upload className="w-6 h-6 text-teal-400" />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-white">Upload Dataset</h2>
              <p className="text-sm text-slate-400">
                Upload your gene expression data
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

        <form onSubmit={handleSubmit} className="p-6 space-y-6">
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">
              Dataset Name
            </label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              required
              className="w-full px-4 py-3 bg-slate-700/50 border border-slate-600/50 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-teal-500/50 focus:border-teal-500/50"
              placeholder="e.g., Breast Cancer RNA-seq"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">
              Description
            </label>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              rows={3}
              className="w-full px-4 py-3 bg-slate-700/50 border border-slate-600/50 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-teal-500/50 focus:border-teal-500/50 resize-none"
              placeholder="Describe your dataset..."
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">
              Data File
            </label>
            <div
              className={`relative border-2 border-dashed rounded-lg p-8 text-center transition-all ${
                file
                  ? "border-teal-500/50 bg-teal-500/10"
                  : "border-slate-600/50 hover:border-slate-500/50 bg-slate-700/30"
              }`}
            >
              <input
                type="file"
                onChange={handleFileChange}
                accept=".csv,.tsv,.xlsx,.xls"
                required
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
              />
              {file ? (
                <div className="flex items-center justify-center gap-3">
                  <CheckCircle2 className="w-8 h-8 text-teal-400" />
                  <div className="text-left">
                    <div className="text-white font-medium">{file.name}</div>
                    <div className="text-sm text-slate-400">
                      {(file.size / 1024 / 1024).toFixed(2)} MB
                    </div>
                  </div>
                </div>
              ) : (
                <div>
                  <FileText className="w-12 h-12 text-slate-400 mx-auto mb-3" />
                  <div className="text-white font-medium mb-1">
                    Drop your file here or click to browse
                  </div>
                  <div className="text-sm text-slate-400">
                    Supports CSV, TSV, and Excel files
                  </div>
                </div>
              )}
            </div>
          </div>

          <div className="flex gap-3 pt-4">
            <button
              type="button"
              onClick={onClose}
              className="flex-1 px-6 py-3 bg-slate-700/50 hover:bg-slate-700 text-white rounded-lg font-medium transition-colors border border-slate-600/50"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={uploading}
              className="flex-1 px-6 py-3 bg-gradient-to-r from-teal-600 to-blue-600 hover:from-teal-500 hover:to-blue-500 text-white rounded-lg font-medium transition-all shadow-lg shadow-teal-500/20 hover:shadow-teal-500/40 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {uploading ? "Uploading..." : "Upload Dataset"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
