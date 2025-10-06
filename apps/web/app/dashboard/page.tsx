"use client";
import { useState } from 'react';
import { Sidebar } from './components/Sidebar';
import { ExperimentDetails } from './components/ExperimentDetails';
import { NewExperimentForm } from './components/NewExperimentForm';
import { EmptyState } from './components/EmptyState';

function App() {
  const [selectedExperimentId, setSelectedExperimentId] = useState<string | null>(null);
  const [showNewExperimentForm, setShowNewExperimentForm] = useState(false);
  const [refreshKey, setRefreshKey] = useState(0);

  const handleNewExperiment = () => {
    setShowNewExperimentForm(true);
  };

  const handleExperimentCreated = () => {
    setRefreshKey(prev => prev + 1);
  };

  return (
    <div className="h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 flex overflow-hidden">
      <Sidebar
        key={refreshKey}
        selectedExperimentId={selectedExperimentId}
        onSelectExperiment={setSelectedExperimentId}
        onNewExperiment={handleNewExperiment}
      />

      <main className="flex-1 overflow-hidden">
        {selectedExperimentId ? (
          <ExperimentDetails experimentId={selectedExperimentId} />
        ) : (
          <EmptyState onNewExperiment={handleNewExperiment} />
        )}
      </main>

      {showNewExperimentForm && (
        <NewExperimentForm
          onClose={() => setShowNewExperimentForm(false)}
          onSuccess={handleExperimentCreated}
        />
      )}
    </div>
  );
}

export default App;
