import { useState } from "react";

const EXAMPLES = [
  {
    label: "Low risk",
    text: "Had a wonderful day out with friends. The weather was amazing and we enjoyed a great lunch together. Feeling grateful for the people in my life.",
  },
  {
    label: "Medium risk",
    text: "Haven't been sleeping well lately. I keep worrying about everything and can't seem to shake this feeling of emptiness. Nothing really excites me anymore and I feel disconnected from people around me.",
  },
  {
    label: "High risk",
    text: "I don't see the point in anything anymore. Every day feels like a struggle and I'm so tired of fighting. I feel completely alone and I just want the pain to stop. Nobody would even notice if I disappeared.",
  },
];

export default function TextInput({ onSubmit, loading }) {
  const [text, setText] = useState("");

  const charCount = text.length;
  const isValid = charCount >= 10 && charCount <= 5000;

  function handleSubmit(e) {
    e.preventDefault();
    if (isValid && !loading) onSubmit(text);
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div>
        <label
          htmlFor="input-text"
          className="block text-sm font-semibold text-gray-700 mb-1"
        >
          Enter social media text for analysis
        </label>
        <textarea
          id="input-text"
          rows={5}
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Paste a tweet, post, or comment here…"
          className="w-full rounded-xl border border-gray-300 px-4 py-3 text-gray-800 shadow-sm
                     placeholder:text-gray-400 focus:border-indigo-500 focus:ring-2 focus:ring-indigo-200
                     transition resize-none"
        />
        <div className="mt-1 flex justify-between text-xs text-gray-500">
          <span>Min 10 characters</span>
          <span className={charCount > 5000 ? "text-red-500 font-medium" : ""}>
            {charCount} / 5 000
          </span>
        </div>
      </div>

      {/* Example buttons */}
      <div className="flex flex-wrap gap-2">
        <span className="text-xs text-gray-500 self-center">Try:</span>
        {EXAMPLES.map((ex) => (
          <button
            key={ex.label}
            type="button"
            onClick={() => setText(ex.text)}
            className="rounded-full border border-gray-300 px-3 py-1 text-xs text-gray-600
                       hover:bg-indigo-50 hover:border-indigo-300 transition cursor-pointer"
          >
            {ex.label}
          </button>
        ))}
      </div>

      <button
        type="submit"
        disabled={!isValid || loading}
        className="w-full rounded-xl bg-indigo-600 py-3 font-semibold text-white shadow
                   hover:bg-indigo-700 focus:ring-2 focus:ring-indigo-300 transition
                   disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer flex items-center justify-center gap-2"
      >
        {loading ? (
          <>
            <svg
              className="h-5 w-5 animate-spin"
              viewBox="0 0 24 24"
              fill="none"
            >
              <circle
                className="opacity-25"
                cx="12"
                cy="12"
                r="10"
                stroke="currentColor"
                strokeWidth="4"
              />
              <path
                className="opacity-75"
                fill="currentColor"
                d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"
              />
            </svg>
            Analyzing…
          </>
        ) : (
          "Analyze Text"
        )}
      </button>
    </form>
  );
}
