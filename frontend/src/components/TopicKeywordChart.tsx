import React from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { Tag } from 'lucide-react';

export interface KeywordWeight {
  keyword: string;
  weight: number;
}

interface TopicKeywordChartProps {
  topicName: string;
  keywords: KeywordWeight[];
}

export const TopicKeywordChart: React.FC<TopicKeywordChartProps> = ({ topicName, keywords }) => {
  return (
    <div className="bg-[#080B16] border border-slate-800/80 rounded-lg p-4 space-y-3">
      <div className="flex items-center justify-between">
        <h4 className="text-sm font-medium text-slate-200 flex items-center gap-2">
          <Tag className="w-4 h-4 text-cyan-400" />
          Top Keyword Frequency Weights — <span className="text-violet-400">{topicName}</span>
        </h4>
        <span className="text-xs text-slate-500">TF-IDF / c-TF-IDF Score</span>
      </div>

      <div className="h-48 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={keywords} layout="vertical" margin={{ top: 5, right: 30, left: 40, bottom: 5 }}>
            <XAxis type="number" stroke="#64748B" fontSize={11} tickLine={false} />
            <YAxis 
              type="category" 
              dataKey="keyword" 
              stroke="#94A3B8" 
              fontSize={12} 
              tickLine={false}
              width={90}
            />
            <Tooltip 
              contentStyle={{ 
                backgroundColor: '#0D1221', 
                borderColor: '#1E293B',
                borderRadius: '0.375rem',
                color: '#F8FAFC',
                fontSize: '12px'
              }}
              formatter={(value: any) => [
                typeof value === 'number' ? value.toFixed(4) : value, 
                'Score'
              ]}
            />
            <Bar dataKey="weight" radius={[0, 4, 4, 0]}>
              {keywords.map((_, index) => (
                <Cell 
                  key={`kw-cell-${index}`} 
                  fill={`hsl(${260 + index * 12}, 85%, 65%)`} 
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};