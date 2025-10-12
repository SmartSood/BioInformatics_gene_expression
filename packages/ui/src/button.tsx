"use client";

import { ReactNode } from "react";

interface ButtonProps {
  children: ReactNode;
  className?: string;
  onClick?: () => void;
  type: "primary"|"secondary";
}
const primaryStyles = "inline-flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-teal-600 to-blue-600 hover:from-teal-500 hover:to-blue-500 text-white rounded-lg font-medium transition-all shadow-lg shadow-teal-500/20 hover:shadow-teal-500/40";
const secondaryStyles = "inline-flex items-center gap-2 px-6 py-3 bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white rounded-lg font-medium transition-all border border-slate-700/50";

export const Button = ({ children, className, onClick, type }: ButtonProps) => {
  return (
    <button
      className={`${type === "primary" ? primaryStyles : secondaryStyles} ${className}`}
      onClick={onClick}

    >
      {children}
    </button>
  );
};