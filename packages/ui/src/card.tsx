"use client";

import { ReactNode } from "react";

type ColorType = 
  | 'slate' | 'gray' | 'zinc' | 'neutral' | 'stone'  // Grays
  | 'red' | 'rose' | 'pink' | 'fuchsia'              // Reds/Pinks
  | 'purple' | 'violet' | 'indigo'                   // Purples
  | 'blue' | 'sky' | 'cyan'                          // Blues
  | 'teal' | 'emerald' | 'green' | 'lime'            // Greens
  | 'yellow' | 'amber' | 'orange';                   // Yellows/Oranges

interface CardProps {
  children: ReactNode;
  title?: string;
  icon?: ReactNode;
  color?: ColorType;
  iconColor?: ColorType;
  className?: string;
}

export const Card = ({ 
  children, 
  title, 
  icon, 
  color = 'slate',
  iconColor, 
  className = "" 
}: CardProps) => {
  const cardColorClasses: Record<ColorType, string> = {
    // Grays
    slate: 'bg-gradient-to-br from-slate-800/90 to-slate-900/90 border-slate-700/50',
    gray: 'bg-gradient-to-br from-gray-800/90 to-gray-900/90 border-gray-700/50',
    zinc: 'bg-gradient-to-br from-zinc-800/90 to-zinc-900/90 border-zinc-700/50',
    neutral: 'bg-gradient-to-br from-neutral-800/90 to-neutral-900/90 border-neutral-700/50',
    stone: 'bg-gradient-to-br from-stone-800/90 to-stone-900/90 border-stone-700/50',
    
    // Reds/Pinks
    red: 'bg-gradient-to-br from-red-800/90 to-red-900/90 border-red-700/50',
    rose: 'bg-gradient-to-br from-rose-800/90 to-rose-900/90 border-rose-700/50',
    pink: 'bg-gradient-to-br from-pink-800/90 to-pink-900/90 border-pink-700/50',
    fuchsia: 'bg-gradient-to-br from-fuchsia-800/90 to-fuchsia-900/90 border-fuchsia-700/50',
    
    // Purples
    purple: 'bg-gradient-to-br from-purple-800/90 to-purple-900/90 border-purple-700/50',
    violet: 'bg-gradient-to-br from-violet-800/90 to-violet-900/90 border-violet-700/50',
    indigo: 'bg-gradient-to-br from-indigo-800/90 to-indigo-900/90 border-indigo-700/50',
    
    // Blues
    blue: 'bg-gradient-to-br from-blue-800/90 to-blue-900/90 border-blue-700/50',
    sky: 'bg-gradient-to-br from-sky-800/90 to-sky-900/90 border-sky-700/50',
    cyan: 'bg-gradient-to-br from-cyan-800/90 to-cyan-900/90 border-cyan-700/50',
    
    // Greens
    teal: 'bg-gradient-to-br from-teal-800/90 to-teal-900/90 border-teal-700/50',
    emerald: 'bg-gradient-to-br from-emerald-800/90 to-emerald-900/90 border-emerald-700/50',
    green: 'bg-gradient-to-br from-green-800/90 to-green-900/90 border-green-700/50',
    lime: 'bg-gradient-to-br from-lime-800/90 to-lime-900/90 border-lime-700/50',
    
    // Yellows/Oranges
    yellow: 'bg-gradient-to-br from-yellow-800/90 to-yellow-900/90 border-yellow-700/50',
    amber: 'bg-gradient-to-br from-amber-800/90 to-amber-900/90 border-amber-700/50',
    orange: 'bg-gradient-to-br from-orange-800/90 to-orange-900/90 border-orange-700/50',
  };

  const iconColorClasses: Record<ColorType, string> = {
    // Grays
    slate: 'bg-slate-500/20 text-slate-400',
    gray: 'bg-gray-500/20 text-gray-400',
    zinc: 'bg-zinc-500/20 text-zinc-400',
    neutral: 'bg-neutral-500/20 text-neutral-400',
    stone: 'bg-stone-500/20 text-stone-400',
    
    // Reds/Pinks
    red: 'bg-red-500/20 text-red-400',
    rose: 'bg-rose-500/20 text-rose-400',
    pink: 'bg-pink-500/20 text-pink-400',
    fuchsia: 'bg-fuchsia-500/20 text-fuchsia-400',
    
    // Purples
    purple: 'bg-purple-500/20 text-purple-400',
    violet: 'bg-violet-500/20 text-violet-400',
    indigo: 'bg-indigo-500/20 text-indigo-400',
    
    // Blues
    blue: 'bg-blue-500/20 text-blue-400',
    sky: 'bg-sky-500/20 text-sky-400',
    cyan: 'bg-cyan-500/20 text-cyan-400',
    
    // Greens
    teal: 'bg-teal-500/20 text-teal-400',
    emerald: 'bg-emerald-500/20 text-emerald-400',
    green: 'bg-green-500/20 text-green-400',
    lime: 'bg-lime-500/20 text-lime-400',
    
    // Yellows/Oranges
    yellow: 'bg-yellow-500/20 text-yellow-400',
    amber: 'bg-amber-500/20 text-amber-400',
    orange: 'bg-orange-500/20 text-orange-400',
  };

  // Use iconColor if provided, otherwise fall back to card color
  const effectiveIconColor = iconColor || color;

  return (
    <div className={`${cardColorClasses[color]} rounded-xl border p-6 shadow-xl ${className}`}>
      {(title || icon) && (
        <div className="flex items-center gap-3 mb-6">
          {icon && (
            <div className={`p-2 ${iconColorClasses[effectiveIconColor]} rounded-lg`}>
              {icon}
            </div>
          )}
          {title && (
            <h2 className="text-xl font-bold text-white">{title}</h2>
          )}
        </div>
      )}
      {children}
    </div>
  );
};
