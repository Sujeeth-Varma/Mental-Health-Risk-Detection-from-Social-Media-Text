const clamp = (v) => Math.max(-1, Math.min(1, v));

export default function SentimentGauge({ sentiment }) {
  if (!sentiment) return null;

  const vader = sentiment.vader ?? {};
  const textblob = sentiment.textblob ?? {};
  const compound = vader.compound ?? 0;

  // Map compound (-1 … 1) to percentage (0 … 100)
  const pct = ((clamp(compound) + 1) / 2) * 100;

  const label =
    compound >= 0.05 ? "Positive" : compound <= -0.05 ? "Negative" : "Neutral";
  const labelColor =
    compound >= 0.05
      ? "text-green-600"
      : compound <= -0.05
        ? "text-red-600"
        : "text-gray-600";

  return (
    <div className="space-y-4">
      <h3 className="text-sm font-semibold text-gray-600 uppercase tracking-wide">
        Sentiment Analysis
      </h3>

      {/* Gauge bar */}
      <div>
        <div className="flex justify-between text-[11px] text-gray-400 mb-1">
          <span>Negative</span>
          <span>Neutral</span>
          <span>Positive</span>
        </div>
        <div className="relative h-4 rounded-full bg-gradient-to-r from-red-400 via-gray-300 to-green-400 overflow-hidden">
          <div
            className="absolute top-0 h-full w-1 bg-white border border-gray-700 rounded-full shadow-md transition-all duration-700"
            style={{ left: `calc(${pct}% - 2px)` }}
          />
        </div>
        <p className={`mt-1 text-center text-lg font-bold ${labelColor}`}>
          {label} ({compound.toFixed(3)})
        </p>
      </div>

      {/* Detailed scores */}
      <div className="grid grid-cols-2 gap-3 text-sm">
        <ScoreCard
          label="VADER Positive"
          value={vader.positive}
          color="text-green-600"
        />
        <ScoreCard
          label="VADER Negative"
          value={vader.negative}
          color="text-red-600"
        />
        <ScoreCard
          label="VADER Neutral"
          value={vader.neutral}
          color="text-gray-600"
        />
        <ScoreCard
          label="VADER Compound"
          value={vader.compound}
          color="text-indigo-600"
        />
        <ScoreCard
          label="Polarity"
          value={textblob.polarity}
          color="text-purple-600"
        />
        <ScoreCard
          label="Subjectivity"
          value={textblob.subjectivity}
          color="text-orange-600"
        />
      </div>
    </div>
  );
}

function ScoreCard({ label, value, color }) {
  return (
    <div className="rounded-lg bg-gray-50 px-3 py-2">
      <p className="text-[11px] text-gray-400">{label}</p>
      <p className={`text-base font-semibold ${color}`}>
        {value != null ? value.toFixed(4) : "—"}
      </p>
    </div>
  );
}
