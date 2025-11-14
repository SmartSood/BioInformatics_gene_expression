"use client";
import { useState, useEffect, use } from "react";
import { Sidebar } from "./components/Sidebar";
import { ExperimentDetails } from "./components/ExperimentDetails";
import { NewExperimentForm } from "./components/NewExperimentForm";
import { EmptyState } from "./components/EmptyState";
import { DatasetSelectionModal } from "./components/DataSelectionUploadModal";
import { DatasetUploadModal } from "./components/DatasetUploadModal";
import { useRouter } from "next/navigation";
import {MODEL_BACKEND_URL} from '@repo/config';
import axios from "axios";
export default function Dashboard() {
  const [userId, setUserId] = useState<string | null>(null);
  const [selectedExperimentId, setSelectedExperimentId] = useState<
    string | null
  >(null);
  const [showDatasetSelection, setShowDatasetSelection] = useState(false);
  const [showDatasetUpload, setShowDatasetUpload] = useState(false);
  const [showNewExperimentForm, setShowNewExperimentForm] = useState(false);
  const [selectedDatasetId, setSelectedDatasetId] = useState<string | null>(
    null
  );
  const [refreshKey, setRefreshKey] = useState(0);
  const router = useRouter();

  useEffect(() => {
    const storedUserId = sessionStorage.getItem("userId");
    const authToken = sessionStorage.getItem("authToken");
    if (storedUserId && authToken) {
      setUserId(storedUserId);
    } else {
      router.push("/login");
    }
  }, [router]);

  const handleLogin = (id: string) => {
    setUserId(id);
  };
  // fetch initial datasets on mount
  
useEffect(() => {
    const fetchDatasets = async () => {
      try {
        const data = await axios.get(`${MODEL_BACKEND_URL}/dataset`, {
          headers: {
            Authorization: `Bearer ${sessionStorage.getItem("authToken")}`,
          },
        });
        //@ts-ignore
        sessionStorage.setItem("datasets", JSON.stringify(data.data.datasets));
      } catch (error) {
        console.error("Failed to fetch datasets", error);
      }
    };
    void fetchDatasets();
  }, []);

  const handleLogout = () => {
    sessionStorage.removeItem("authToken");
    sessionStorage.removeItem("userId");
    sessionStorage.removeItem("User");
    setUserId(null);
    setSelectedExperimentId(null);
  };

  const handleNewExperiment = () => {
    setShowDatasetSelection(true);
  };

  const handleDatasetSelected = (datasetId: string) => {
    setSelectedDatasetId(datasetId);
    setShowDatasetSelection(false);
    setShowNewExperimentForm(true);
  };

  const handleUploadNew = () => {
    setShowDatasetSelection(false);
    setShowDatasetUpload(true);
  };

  const handleUploadComplete = (datasetId: string) => {
    setSelectedDatasetId(datasetId);
    setShowDatasetUpload(false);
    setShowNewExperimentForm(true);
  };

  const handleExperimentCreated = () => {
    setRefreshKey((prev) => prev + 1);
    setSelectedDatasetId(null);
  };

  return (
    <div className="h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 flex overflow-hidden">
      <Sidebar
        key={refreshKey}
        selectedExperimentId={selectedExperimentId}
        onSelectExperiment={setSelectedExperimentId}
        onNewExperiment={handleNewExperiment}
        onLogout={handleLogout}
      />

      <main className="flex-1 overflow-hidden">
        {selectedExperimentId ? (
          <ExperimentDetails experimentId={selectedExperimentId} />
        ) : (
          <EmptyState onNewExperiment={handleNewExperiment} />
        )}
      </main>

      {showDatasetSelection && (
        <DatasetSelectionModal
          onClose={() => setShowDatasetSelection(false)}
          onSelectDataset={handleDatasetSelected}
          onUploadNew={handleUploadNew}
        />
      )}

      {showDatasetUpload && (
        <DatasetUploadModal
          onClose={() => setShowDatasetUpload(false)}
          onUploadComplete={handleUploadComplete}
        />
      )}

      {showNewExperimentForm && selectedDatasetId && (
        <NewExperimentForm
          datasetId={selectedDatasetId}
          onClose={() => {
            setShowNewExperimentForm(false);
            setSelectedDatasetId(null);
          }}
          onSuccess={handleExperimentCreated}
        />
      )}
    </div>
  );
}
