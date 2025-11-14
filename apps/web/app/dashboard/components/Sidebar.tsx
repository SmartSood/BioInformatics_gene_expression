import { useExperiments } from "../../../hooks/useExperiment";
import {
  Plus,
  Dna,
  Clock,
  CheckCircle,
  XCircle,
  Loader,
  LogOut,
  User,
} from "lucide-react";
import { Experiment } from "../../../utils/scemma";
import { useState, useEffect } from "react";
import { Button } from "@repo/ui/button";
import { SideCard } from "@repo/ui/sidecard";

interface SidebarProps {
  selectedExperimentId: string | null;
  onSelectExperiment: (id: string) => void;
  onNewExperiment: () => void;
  onLogout: () => void;
}

export function Sidebar({
  selectedExperimentId,
  onSelectExperiment,
  onNewExperiment,
  onLogout,
}: SidebarProps) {
  const { experiments, loading } = useExperiments();
  const [userName, setUserName] = useState<string>("User");
  const [userEmail, setUserEmail] = useState<string>("");

  useEffect(() => {
    const currentUser = sessionStorage.getItem("User");
    console.log("Current User from sessionStorage:", currentUser);
    if (currentUser) {
      const user = JSON.parse(currentUser);
      setUserName(user.name || "User");
      setUserEmail(user.email || "");
    }
  }, []);

  const getStatusIcon = (status: Experiment["status"]) => {
    switch (status) {
      case "finished":
        return <CheckCircle className="w-4 h-4 text-emerald-400" />;
      case "started":
        return <Loader className="w-4 h-4 text-blue-400 animate-spin" />;
      case "failed":
        return <XCircle className="w-4 h-4 text-red-400" />;
      default:
        return <Clock className="w-4 h-4 text-amber-400" />;
    }
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return new Intl.DateTimeFormat("en-US", {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    }).format(date);
  };

  return (
    <div className="w-80 bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 border-r border-slate-700/50 flex flex-col h-screen">
      <div className="p-6 border-b border-slate-700/50">
        <div className="flex items-center gap-3 mb-4">
          <div className="p-2 bg-gradient-to-br from-teal-500/20 to-blue-500/20 rounded-lg">
            <Dna className="w-6 h-6 text-teal-400" />
          </div>
          <div>
            <h1 className="text-xl font-bold text-white">DrugTarget AI</h1>
            <p className="text-xs text-slate-400">Interaction Analysis</p>
          </div>
        </div>

        <div className="mb-6 p-3 bg-slate-800/50 rounded-lg border border-slate-700/50">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-full bg-gradient-to-br from-teal-500 to-blue-500 flex items-center justify-center flex-shrink-0">
              <User className="w-5 h-5 text-white" />
            </div>
            <div className="overflow-hidden">
              <div className="text-sm font-semibold text-white truncate">
                {userName}
              </div>
              <div className="text-xs text-slate-400 truncate">{userEmail}</div>
            </div>
          </div>
        </div>

        <Button
          onClick={onNewExperiment}
          type="primary"
          className="w-full justify-center"
        >
          <Plus className="w-5 h-5" />
          New Analysis
        </Button>
      </div>

      <div className="flex-1 overflow-y-auto p-4">
        <h2 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3 px-2">
          Recent Experiments
        </h2>

        {loading ? (
          <div className="flex items-center justify-center py-8">
            <Loader className="w-6 h-6 text-teal-400 animate-spin" />
          </div>
        ) : experiments.length === 0 ? (
          <div className="text-center py-8 px-4">
            <p className="text-slate-500 text-sm">No experiments yet</p>
            <p className="text-slate-600 text-xs mt-1">
              Create your first analysis
            </p>
          </div>
        ) : (
          <div className="space-y-2">
            {experiments.map((experiment) => (
              <SideCard
                key={experiment.id}
                title={experiment.name}
                description={experiment.description}
                statusIcon={getStatusIcon(experiment.status)}
                createdAt={experiment.updatedAt}
                onClick={() => onSelectExperiment(experiment.id)}
                selected={selectedExperimentId === experiment.id}
              />
            ))}
          </div>
        )}
      </div>

      <div className="p-4 border-t border-slate-700/50">
        <Button
          onClick={onLogout}
          type="secondary"
          className="w-full justify-center"
        >
          <LogOut className="w-5 h-5" />
          Logout
        </Button>
      </div>
    </div>
  );
}
