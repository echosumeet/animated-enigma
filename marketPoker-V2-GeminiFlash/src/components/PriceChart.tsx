import React from 'react';
import { MarketPricePoint } from '../types/game';

interface PriceChartProps {
  priceHistory: MarketPricePoint[];
  height?: number;
}

export const PriceChart: React.FC<PriceChartProps> = ({ priceHistory, height = 120 }) => {
  if (!priceHistory || priceHistory.length === 0) {
    return (
      <div
        style={{ height }}
        className="w-full flex items-center justify-center text-xs text-slate-500 font-mono"
      >
        Awaiting price discovery trades...
      </div>
    );
  }

  const width = 450;
  const padding = 20;

  const minPrice = 0.0;
  const maxPrice = 1.0;

  const points = priceHistory.map((pt, idx) => {
    const x = padding + (idx / Math.max(1, priceHistory.length - 1)) * (width - 2 * padding);
    const y = height - padding - ((pt.price - minPrice) / (maxPrice - minPrice)) * (height - 2 * padding);
    return { x, y, price: pt.price, phase: pt.phase };
  });

  const pathD = points.reduce((acc, curr, idx) => {
    return idx === 0 ? `M ${curr.x} ${curr.y}` : `${acc} L ${curr.x} ${curr.y}`;
  }, '');

  const areaD = points.length > 0
    ? `${pathD} L ${points[points.length - 1].x} ${height - padding} L ${points[0].x} ${height - padding} Z`
    : '';

  return (
    <div className="w-full relative">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="w-full overflow-visible font-mono select-none"
      >
        {/* Horizontal Gridlines */}
        {[0.25, 0.5, 0.75].map((level) => {
          const y = height - padding - level * (height - 2 * padding);
          return (
            <g key={level}>
              <line
                x1={padding}
                y1={y}
                x2={width - padding}
                y2={y}
                stroke="#334155"
                strokeDasharray="3 3"
                strokeWidth="0.75"
              />
              <text x={padding - 2} y={y + 3} fill="#64748b" fontSize="8" textAnchor="end">
                {level * 100}%
              </text>
            </g>
          );
        })}

        {/* Gradient fill underneath */}
        <defs>
          <linearGradient id="chartGradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#6366f1" stopOpacity="0.35" />
            <stop offset="100%" stopColor="#6366f1" stopOpacity="0.0" />
          </linearGradient>
        </defs>

        {areaD && <path d={areaD} fill="url(#chartGradient)" />}

        {/* Price Line */}
        {pathD && (
          <path
            d={pathD}
            fill="none"
            stroke="#818cf8"
            strokeWidth="2.5"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        )}

        {/* Phase / Trade Dots */}
        {points.map((pt, i) => (
          <circle
            key={i}
            cx={pt.x}
            cy={pt.y}
            r="3.5"
            className="fill-indigo-400 stroke-slate-950 stroke-2 hover:r-5 transition-all"
          >
            <title>{`Price: $${pt.price.toFixed(2)} (${pt.phase})`}</title>
          </circle>
        ))}
      </svg>
    </div>
  );
};
