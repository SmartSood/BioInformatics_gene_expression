"use client";

import { ReactNode } from "react";
import { Clock } from "lucide-react";

interface SideCardProps {
  title: string;
  description?: string;
  statusIcon: ReactNode;
  createdAt: string;
  onClick?: () => void;
  selected?: boolean;
  className?: string;
}

export const SideCard = ({ 
  title, 
  description, 
  statusIcon, 
  createdAt, 
  onClick, 
  selected = false, 
  className = "" 
}: SideCardProps) => {
  const formatDate = (dateString: string) => {
    const cleaned = dateString.replace(/(\.\d{3})\d+Z$/, "$1Z");
    
    return new Date(cleaned).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  return (
    <button
      onClick={onClick}
      className={`w-full text-left p-4 rounded-lg transition-all ${
        selected
          ? 'bg-gradient-to-r from-teal-600/30 to-blue-600/30 border border-teal-500/50 shadow-lg'
          : 'bg-slate-800/50 hover:bg-slate-800 border border-transparent'
      } ${className}`}
    >
      <div className="flex items-start justify-between mb-2">
        <h3 className="font-semibold text-white text-sm line-clamp-1">
          {title}
        </h3>
        {statusIcon}
      </div>

      {description && (
        <p className="text-xs text-slate-400 mb-2 line-clamp-2">
          {description}
        </p>
      )}

      <div className="flex items-center gap-2 text-xs text-slate-500">
        <Clock className="w-3 h-3" />
        {formatDate(createdAt)}
      </div>
    </button>
  );
};