import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";

const COLORS = [
  "#6366f1",
  "#8b5cf6",
  "#a855f7",
  "#d946ef",
  "#ec4899",
  "#f43f5e",
  "#f97316",
  "#eab308",
  "#22c55e",
  "#14b8a6",
];

export default function FeatureChart({ emotions, topic }) {
  if (!emotions) return null;

  // Build data for emotion chart – filter out zero values
  const emotionData = Object.entries(emotions)
    .filter(([, v]) => v > 0)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 10)
    .map(([name, value]) => ({ name, value: +(value * 100).toFixed(1) }));

  return (
    <div className="space-y-6">
      {/* Emotion bar chart */}
      {emotionData.length > 0 && (
        <div>
          <h3 className="mb-3 text-sm font-semibold text-gray-600 uppercase tracking-wide">
            Emotional Tone
          </h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={emotionData}
                layout="vertical"
                margin={{ left: 70 }}
              >
                <XAxis
                  type="number"
                  domain={[0, 100]}
                  tick={{ fontSize: 11 }}
                  unit="%"
                />
                <YAxis
                  type="category"
                  dataKey="name"
                  tick={{ fontSize: 12, fill: "#4b5563" }}
                  width={65}
                />
                <Tooltip
                  formatter={(v) => `${v}%`}
                  contentStyle={{ borderRadius: 8, fontSize: 13 }}
                />
                <Bar dataKey="value" radius={[0, 6, 6, 0]} barSize={18}>
                  {emotionData.map((_, idx) => (
                    <Cell key={idx} fill={COLORS[idx % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Topic keywords */}
      {topic && topic.keywords && topic.keywords.length > 0 && (
        <div>
          <h3 className="mb-2 text-sm font-semibold text-gray-600 uppercase tracking-wide">
            Detected Topic
          </h3>
          <div className="flex flex-wrap gap-2">
            {topic.keywords.map((kw, i) => (
              <span
                key={i}
                className="rounded-full bg-indigo-100 px-3 py-1 text-xs font-medium text-indigo-700"
              >
                {kw}
              </span>
            ))}
          </div>
          {topic.confidence != null && (
            <p className="mt-1 text-[11px] text-gray-400">
              Confidence: {(topic.confidence * 100).toFixed(1)}%
            </p>
          )}
        </div>
      )}
    </div>
  );
}
