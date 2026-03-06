"use client";
import { useState } from "react";
import Uploader from "@/components/Uploader";
import ResultsPanel from "@/components/ResultsPanel";
import { AnalysisResult, Settings } from "@/lib/types";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

const N_OPTIONS = [2, 4, 6, 8, 10, 12, 15, 20, 25, 30];

export default function Home() {
  const [testFile, setTestFile]   = useState<File | null>(null);
  const [settings, setSettings]   = useState<Settings>({ nList: "10,15,20,25,30", cvFolds: 5, useGridSearch: false });
  const [loading, setLoading]     = useState(false);
  const [error, setError]         = useState<string | null>(null);
  const [result, setResult]       = useState<AnalysisResult | null>(null);
  const [progress, setProgress]   = useState<string>("");

  // Toggle an N value in the comma list
  const toggleN = (n: number) => {
    const current = settings.nList.split(",").map((x) => x.trim()).filter(Boolean).map(Number);
    const updated = current.includes(n) ? current.filter((x) => x !== n) : [...current, n].sort((a, b) => a - b);
    setSettings((s) => ({ ...s, nList: updated.join(",") }));
  };

  const selectedNs = settings.nList.split(",").map((x) => parseInt(x.trim())).filter(Boolean);

  const handleRun = async () => {
    if (selectedNs.length === 0) { setError("Select at least one N value."); return; }
    setLoading(true);
    setError(null);
    setResult(null);
    setProgress("Sending data to backend…");

    try {
      const form = new FormData();
      form.append("n_list",          settings.nList);
      form.append("cv_folds",        String(settings.cvFolds));
      form.append("use_grid_search", String(settings.useGridSearch));
      if (testFile) form.append("test_file", testFile);

      setProgress("Running ML pipeline (this may take a minute)…");
      const resp = await fetch(`${API_URL}/api/analyze`, { method: "POST", body: form });

      if (!resp.ok) {
        const data = await resp.json().catch(() => ({}));
        throw new Error(data.detail ?? `Server error ${resp.status}`);
      }

      const data: AnalysisResult = await resp.json();
      setResult(data);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setLoading(false);
      setProgress("");
    }
  };

  return (
    <div className="flex min-h-screen">
      {/* ── Sidebar ───────────────────────────────────────────────── */}
      <aside className="hidden lg:flex flex-col w-72 bg-white/[0.04] border-r border-white/[0.08] p-6 gap-6 sticky top-0 h-screen overflow-y-auto">
        <div>
          <h1 className="text-xl font-bold text-white tracking-tight">🧬 Disease Predictor</h1>
          <p className="text-white/40 text-xs mt-1">Gene expression microarray classifier</p>
        </div>

        <hr className="border-white/10" />

        <div>
          <p className="text-xs font-bold uppercase tracking-widest text-purple-300 mb-3">ℹ️ About</p>
          <p className="text-white/60 text-xs leading-relaxed">
            Classifies brain tumours from gene expression data into one of five disease types:
          </p>
          <ul className="mt-2 space-y-1 text-xs">
            {[["🔴", "EPD", "Ependymoma"],["🟡","JPA","Juvenile Pilocytic Astrocytoma"],["🔵","MED","Medulloblastoma"],["🟢","MGL","Malignant Glioma"],["🟣","RHB","Rhabdoid Tumour"]].map(([icon, code, name]) => (
              <li key={code} className="text-white/50"><span className="mr-1">{icon}</span><span className="font-semibold text-white/70">{code}</span> – {name}</li>
            ))}
          </ul>
        </div>

        <hr className="border-white/10" />

        <div className="space-y-4">
          <p className="text-xs font-bold uppercase tracking-widest text-purple-300">⚙️ Model Settings</p>

          {/* N-gene selector */}
          <div>
            <label className="text-xs text-white/50 mb-2 block">Top-N gene subsets</label>
            <div className="flex flex-wrap gap-1.5">
              {N_OPTIONS.map((n) => (
                <button
                  key={n}
                  onClick={() => toggleN(n)}
                  className={`px-2.5 py-1 rounded-lg text-xs font-semibold transition-all
                    ${selectedNs.includes(n)
                      ? "bg-purple-600/60 border border-purple-400/60 text-purple-100"
                      : "bg-white/[0.06] border border-white/10 text-white/40 hover:border-purple-500/40"}`}
                >
                  {n}
                </button>
              ))}
            </div>
          </div>

          {/* CV folds */}
          <div>
            <label className="text-xs text-white/50 mb-1 block">
              Cross-validation folds: <span className="text-purple-300 font-bold">{settings.cvFolds}</span>
            </label>
            <input
              type="range" min={3} max={10} value={settings.cvFolds}
              onChange={(e) => setSettings((s) => ({ ...s, cvFolds: parseInt(e.target.value) }))}
              className="w-full accent-purple-500"
            />
          </div>

          {/* GridSearch toggle */}
          <label className="flex items-center gap-2 cursor-pointer select-none">
            <div className="relative">
              <input
                type="checkbox"
                checked={settings.useGridSearch}
                onChange={(e) => setSettings((s) => ({ ...s, useGridSearch: e.target.checked }))}
                className="sr-only"
              />
              <div className={`w-10 h-5 rounded-full transition-colors ${settings.useGridSearch ? "bg-purple-600" : "bg-white/20"}`} />
              <div className={`absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white shadow transition-transform ${settings.useGridSearch ? "translate-x-5" : ""}`} />
            </div>
            <span className="text-xs text-white/60">Enable GridSearchCV tuning</span>
          </label>
          {settings.useGridSearch && (
            <p className="text-xs text-yellow-400/80 bg-yellow-500/10 border border-yellow-500/20 rounded-lg px-3 py-2">
              ⚠️ GridSearch is slower and may timeout on the free Render tier.
            </p>
          )}
        </div>

        <div className="mt-auto">
          <p className="text-white/20 text-xs">Built with ❤️ using Next.js &amp; FastAPI</p>
        </div>
      </aside>

      {/* ── Main Content ──────────────────────────────────────────── */}
      <main className="flex-1 px-4 md:px-8 py-8 max-w-5xl mx-auto w-full">
        {/* Hero */}
        <div className="rounded-2xl p-8 mb-8 text-center"
          style={{ background: "linear-gradient(135deg, rgba(139,92,246,0.25), rgba(59,130,246,0.25))", border: "1px solid rgba(139,92,246,0.35)" }}>
          <div className="text-5xl mb-3">🧬</div>
          <h1 className="text-3xl md:text-4xl font-extrabold text-white mb-2">Disease Prediction Model</h1>
          <p className="text-white/55 text-sm">Gene expression microarray classification using machine learning</p>
        </div>

        {/* Mobile settings strip */}
        <div className="lg:hidden glass-card p-4 mb-6 space-y-3">
          <p className="text-xs font-bold uppercase tracking-widest text-purple-300">⚙️ Quick Settings</p>
          <div className="flex flex-wrap gap-1.5">
            {N_OPTIONS.map((n) => (
              <button key={n} onClick={() => toggleN(n)}
                className={`px-2.5 py-1 rounded-lg text-xs font-semibold transition-all
                  ${selectedNs.includes(n)
                    ? "bg-purple-600/60 border border-purple-400/60 text-purple-100"
                    : "bg-white/[0.06] border border-white/10 text-white/40"}`}>
                {n}
              </button>
            ))}
          </div>
          <label className="flex items-center gap-2 cursor-pointer select-none">
            <input type="checkbox" checked={settings.useGridSearch}
              onChange={(e) => setSettings((s) => ({ ...s, useGridSearch: e.target.checked }))}
              className="accent-purple-500 w-4 h-4" />
            <span className="text-xs text-white/60">GridSearchCV tuning</span>
          </label>
        </div>

        {/* Info banner */}
        <div className="rounded-xl px-4 py-3 mb-5 text-sm text-blue-300 bg-blue-500/10 border border-blue-500/25">
          ℹ️ Training data &amp; class labels are loaded automatically from the bundled dataset. Upload your <strong>own test CSV</strong> below, or leave blank to use the bundled sample.
        </div>

        {/* Upload */}
        <div className="mb-6">
          <h2 className="text-sm font-bold uppercase tracking-widest text-purple-300 mb-3">📂 Test Data</h2>
          <Uploader file={testFile} onFile={setTestFile} />
        </div>

        {/* Run button */}
        <button
          id="run-analysis-btn"
          onClick={handleRun}
          disabled={loading || selectedNs.length === 0}
          className="w-full md:w-auto px-8 py-3.5 rounded-xl font-bold text-white text-sm transition-all
            bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-500 hover:to-blue-500
            shadow-lg shadow-purple-900/40 hover:shadow-purple-700/40
            disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {loading ? (
            <span className="flex items-center gap-2">
              <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
              </svg>
              Running Analysis…
            </span>
          ) : "🚀 Run Analysis"}
        </button>

        {/* Progress */}
        {loading && progress && (
          <p className="mt-3 text-sm text-white/50 animate-pulse">{progress}</p>
        )}

        {/* Error */}
        {error && (
          <div className="mt-4 rounded-xl px-4 py-3 text-sm text-red-300 bg-red-500/10 border border-red-500/25">
            ❌ {error}
          </div>
        )}

        {/* Results */}
        {result && (
          <div className="mt-10">
            <hr className="border-white/10 mb-8" />
            <ResultsPanel result={result} />
          </div>
        )}

        {/* Idle state */}
        {!result && !loading && !error && (
          <div className="mt-10 rounded-2xl border border-white/10 bg-white/[0.03] p-10 text-center">
            <div className="text-4xl mb-3">🧬</div>
            <h3 className="text-white/60 font-semibold text-lg">Ready to Analyse</h3>
            <p className="text-white/35 text-sm mt-1">Configure settings in the sidebar, then click <strong>Run Analysis</strong>.</p>
          </div>
        )}
      </main>
    </div>
  );
}
