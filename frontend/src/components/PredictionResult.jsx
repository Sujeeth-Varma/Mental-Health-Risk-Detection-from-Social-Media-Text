const COLORS = { Low: "#22c55e", Medium: "#eab308", High: "#ef4444" };
const BG = { Low: "bg-green-50", Medium: "bg-yellow-50", High: "bg-red-50" };
const BORDER = {
  Low: "border-green-400",
  Medium: "border-yellow-400",
  High: "border-red-400",
};
const EMOJI = { Low: "✅", Medium: "⚠️", High: "🚨" };

export default function PredictionResult({ prediction }) {
  if (!prediction) return null;

  const { risk_level, description, probabilities, model_used } = prediction;
  const color = COLORS[risk_level] ?? "#6b7280";

  return (
    <div
      className={`rounded-2xl border-2 ${BORDER[risk_level]} ${BG[risk_level]} p-6 space-y-4`}
    >
      {/* Badge */}
      <div className="flex items-center gap-3">
        <span className="text-3xl">{EMOJI[risk_level]}</span>
        <div>
          <p className="text-sm font-medium text-gray-500">
            Predicted Risk Level
          </p>
          <p className="text-2xl font-bold" style={{ color }}>
            {risk_level} Risk
          </p>
        </div>
      </div>

      <p className="text-sm text-gray-600">{description}</p>

      {/* Probability bars */}
      {probabilities && Object.keys(probabilities).length > 0 && (
        <div className="space-y-2">
          <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide">
            Confidence
          </p>
          {["low", "medium", "high"].map((key) => {
            const pct = ((probabilities[key] ?? 0) * 100).toFixed(1);
            const barColor =
              key === "low"
                ? "bg-green-500"
                : key === "medium"
                  ? "bg-yellow-500"
                  : "bg-red-500";
            return (
              <div key={key} className="flex items-center gap-2 text-sm">
                <span className="w-16 capitalize text-gray-600">{key}</span>
                <div className="flex-1 h-3 rounded-full bg-gray-200 overflow-hidden">
                  <div
                    className={`h-full rounded-full ${barColor} transition-all duration-700`}
                    style={{ width: `${pct}%` }}
                  />
                </div>
                <span className="w-14 text-right font-medium text-gray-700">
                  {pct}%
                </span>
              </div>
            );
          })}
        </div>
      )}

      <p className="text-[11px] text-gray-400">Model: {model_used}</p>
    </div>
  );
}
