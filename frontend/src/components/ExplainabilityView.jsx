export default function ExplainabilityView({ explanation, features }) {
  if (!explanation && !features) return null;

  const words = explanation?.word_contributions ?? [];
  const unigrams = features?.top_unigrams ?? [];
  const bigrams = features?.top_bigrams ?? [];

  return (
    <div className="space-y-6">
      {/* Word-level LIME contributions */}
      {words.length > 0 && (
        <Section title="Key Word Contributions (LIME)">
          <div className="flex flex-wrap gap-2">
            {words.map((w, i) => {
              const abs = Math.abs(w.weight);
              const bg =
                w.impact === "positive"
                  ? `rgba(239,68,68,${Math.min(abs * 3, 1)})`
                  : `rgba(34,197,94,${Math.min(abs * 3, 1)})`;
              return (
                <span
                  key={i}
                  title={`weight: ${w.weight}`}
                  className="rounded-full px-3 py-1 text-sm font-medium text-neutral-500 shadow-sm"
                  style={{ backgroundColor: bg }}
                >
                  {w.word}{" "}
                  <span className="opacity-75 text-xs">
                    ({w.weight > 0 ? "+" : ""}
                    {w.weight.toFixed(3)})
                  </span>
                </span>
              );
            })}
          </div>
          <p className="mt-2 text-[11px] text-gray-400">
            <span className="text-red-400">Red</span> = pushes toward higher
            risk &nbsp;|&nbsp; <span className="text-green-500">Green</span> =
            pushes toward lower risk
          </p>
        </Section>
      )}

      {/* Top unigrams */}
      {unigrams.length > 0 && (
        <Section title="Top Unigrams">
          <div className="flex flex-wrap gap-2">
            {unigrams.map((u, i) => (
              <span
                key={i}
                className="rounded-full border border-indigo-200 bg-indigo-50 px-3 py-1 text-xs text-indigo-700"
              >
                {u.term}
                <span className="ml-1 text-indigo-400">×{u.count}</span>
              </span>
            ))}
          </div>
        </Section>
      )}

      {/* Top bigrams */}
      {bigrams.length > 0 && (
        <Section title="Top Bigrams">
          <div className="flex flex-wrap gap-2">
            {bigrams.map((b, i) => (
              <span
                key={i}
                className="rounded-full border border-purple-200 bg-purple-50 px-3 py-1 text-xs text-purple-700"
              >
                {b.term}
                <span className="ml-1 text-purple-400">×{b.count}</span>
              </span>
            ))}
          </div>
        </Section>
      )}
    </div>
  );
}

function Section({ title, children }) {
  return (
    <div>
      <h3 className="mb-2 text-sm font-semibold text-gray-600 uppercase tracking-wide">
        {title}
      </h3>
      {children}
    </div>
  );
}
