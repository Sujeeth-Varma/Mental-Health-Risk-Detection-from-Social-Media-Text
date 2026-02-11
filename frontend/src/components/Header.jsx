export default function Header({ healthy }) {
  return (
    <header className="bg-gradient-to-r from-indigo-700 via-purple-700 to-pink-600 text-white shadow-lg">
      <div className="mx-auto max-w-5xl px-4 py-6 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span className="text-3xl">🧠</span>
          <div>
            <h1 className="text-2xl font-bold tracking-tight">
              Mental Health Risk Detector
            </h1>
            <p className="text-sm text-indigo-200">
              Explainable AI for multi-level risk detection from social media
              text
            </p>
          </div>
        </div>

        <span
          className={`inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-medium ${
            healthy
              ? "bg-green-500/20 text-green-200"
              : "bg-red-500/20 text-red-200"
          }`}
        >
          <span
            className={`h-2 w-2 rounded-full ${
              healthy ? "bg-green-400 animate-pulse" : "bg-red-400"
            }`}
          />
          {healthy ? "API Connected" : "API Offline"}
        </span>
      </div>
    </header>
  );
}
