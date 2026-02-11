import { useEffect, useState } from "react";
import Header from "./components/Header";
import TextInput from "./components/TextInput";
import PredictionResult from "./components/PredictionResult";
import ExplainabilityView from "./components/ExplainabilityView";
import SentimentGauge from "./components/SentimentGauge";
import FeatureChart from "./components/FeatureChart";
import { predictRisk, checkHealth } from "./services/api";

export default function App() {
  const [healthy, setHealthy] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  // Health check on mount
  useEffect(() => {
    checkHealth()
      .then((d) => setHealthy(d.models_loaded === true))
      .catch(() => setHealthy(false));
  }, []);

  async function handleSubmit(text) {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const data = await predictRisk(text);
      setResult(data);
    } catch (err) {
      const msg =
        err.response?.data?.error ?? err.message ?? "Something went wrong.";
      setError(msg);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-gray-50 text-gray-900">
      <Header healthy={healthy} />

      <main className="mx-auto max-w-5xl px-4 py-8 space-y-8">
        {/* Input */}
        <section className="rounded-2xl bg-white p-6 shadow-sm border border-gray-200">
          <TextInput onSubmit={handleSubmit} loading={loading} />
        </section>

        {/* Error */}
        {error && (
          <div className="rounded-xl border border-red-300 bg-red-50 px-4 py-3 text-sm text-red-700">
            {error}
          </div>
        )}

        {/* Results */}
        {result && (
          <div className="space-y-6">
            {/* Prediction badge + probabilities */}
            <section className="rounded-2xl bg-white p-6 shadow-sm border border-gray-200">
              <PredictionResult prediction={result.prediction} />
            </section>

            {/* Two-column: Sentiment / Emotions */}
            <div className="grid gap-6 md:grid-cols-2">
              <section className="rounded-2xl bg-white p-6 shadow-sm border border-gray-200">
                <SentimentGauge
                  sentiment={{
                    vader: result.features?.vader_sentiment,
                    textblob: result.features?.textblob_sentiment,
                  }}
                />
              </section>

              <section className="rounded-2xl bg-white p-6 shadow-sm border border-gray-200">
                <FeatureChart
                  emotions={result.features?.emotional_tone}
                  topic={result.topic}
                />
              </section>
            </div>

            {/* Explainability */}
            <section className="rounded-2xl bg-white p-6 shadow-sm border border-gray-200">
              <ExplainabilityView
                explanation={result.explanation}
                features={result.features}
              />
            </section>
          </div>
        )}

        {/* Footer */}
        <footer className="text-center text-xs text-gray-400 pt-4 pb-8">
          Built for research & education &middot; © {new Date().getFullYear()}
        </footer>
      </main>
    </div>
  );
}
