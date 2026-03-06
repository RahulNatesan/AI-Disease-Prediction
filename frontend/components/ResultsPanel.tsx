"use client";
import { AnalysisResult } from "@/lib/types";
import {
  LineChart, Line, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer,
  PieChart, Pie, Cell, BarChart, Bar,
} from "recharts";

const PALETTE = ["#8b5cf6", "#ec4899", "#3b82f6", "#10b981", "#f59e0b", "#ef4444"];

interface Props {
  result: AnalysisResult;
}

const badgeClass: Record<string, string> = {
  EPD: "badge-EPD",
  JPA: "badge-JPA",
  MED: "badge-MED",
  MGL: "badge-MGL",
  RHB: "badge-RHB",
};

export default function ResultsPanel({ result }: Props) {
  // Build line chart data: x = N genes, series = model error rates
  const lineData = result.n_list.map((n, i) => {
    const point: Record<string, number | string> = { n };
    result.model_names.forEach((name, j) => {
      point[name] = parseFloat((result.error_rates[i][j] * 100).toFixed(2));
    });
    return point;
  });

  // Best accuracy per model (for bar chart)
  const bestPerModel = result.model_names.map((name) => {
    const rows = result.results.filter((r) => r.model === name);
    const best = rows.reduce((a, b) => (a.accuracy > b.accuracy ? a : b), rows[0]);
    return { model: name, accuracy: parseFloat((best.accuracy * 100).toFixed(2)) };
  });

  const displayResults = result.results.map((r) => ({
    ...r,
    accuracy: (r.accuracy * 100).toFixed(2) + "%",
    f1: (r.f1 * 100).toFixed(2) + "%",
    precision: (r.precision * 100).toFixed(2) + "%",
    recall: (r.recall * 100).toFixed(2) + "%",
  }));

  return (
    <div className="space-y-8">
      {/* ── Best Configuration ────────────────────────────────────── */}
      <section>
        <h2 className="text-lg font-bold text-purple-300 border-b border-purple-500/30 pb-2 mb-4">
          🏆 Best Configuration
        </h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {[
            { label: "Best Model",  value: result.best_model, small: true },
            { label: "Best N Genes", value: result.best_n.toString() },
            { label: "Accuracy",   value: (result.best_score * 100).toFixed(1) + "%" },
            { label: "F1 Score",   value: (result.best_f1 * 100).toFixed(1) + "%" },
          ].map(({ label, value, small }) => (
            <div key={label} className="glass-card p-5 text-center hover:-translate-y-0.5 transition-transform">
              <p className="text-xs font-semibold uppercase tracking-widest text-white/50 mb-1">{label}</p>
              <p className={`font-bold text-white leading-tight ${small ? "text-xl" : "text-3xl"}`}>{value}</p>
            </div>
          ))}
        </div>
      </section>

      {/* ── Dataset Summary ───────────────────────────────────────── */}
      <section>
        <h2 className="text-lg font-bold text-purple-300 border-b border-purple-500/30 pb-2 mb-4">
          📊 Training Dataset Distribution
        </h2>
        <div className="flex flex-wrap gap-3">
          {Object.entries(result.class_dist).map(([label, count]) => (
            <div key={label} className="glass-card px-5 py-3 text-center min-w-[100px]">
              <p className="text-xs uppercase tracking-wider text-white/50">{label}</p>
              <p className="text-2xl font-bold text-white">{count}</p>
              <p className="text-xs text-white/35">samples</p>
            </div>
          ))}
        </div>
      </section>

      {/* ── Charts ──────────────────────────────────────────────────── */}
      <section>
        <h2 className="text-lg font-bold text-purple-300 border-b border-purple-500/30 pb-2 mb-4">
          📈 Visualizations
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Error Rate Line Chart */}
          <div className="glass-card p-5">
            <p className="text-sm font-semibold text-white/70 mb-3">Error Rate vs N Genes</p>
            <ResponsiveContainer width="100%" height={240}>
              <LineChart data={lineData} margin={{ top: 5, right: 10, left: -20, bottom: 5 }}>
                <XAxis dataKey="n" stroke="#ffffff50" tick={{ fill: "#ffffff70", fontSize: 11 }} label={{ value: "N Genes", position: "insideBottomRight", fill: "#ffffff50", fontSize: 11 }} />
                <YAxis stroke="#ffffff50" tick={{ fill: "#ffffff70", fontSize: 11 }} tickFormatter={(v) => v + "%"} />
                <Tooltip
                  contentStyle={{ background: "#1e1b4b", border: "1px solid #4c1d95", borderRadius: 8 }}
                  labelStyle={{ color: "#c4b5fd" }}
                  itemStyle={{ color: "#e2e8f0" }}
                  formatter={(v: unknown) => (v as number) + "%"}
                />
                <Legend wrapperStyle={{ fontSize: 11, color: "#ffffffaa" }} />
                {result.model_names.map((name, i) => (
                  <Line key={name} type="monotone" dataKey={name} stroke={PALETTE[i % PALETTE.length]} dot={{ r: 3 }} strokeWidth={2} />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Prediction Pie Chart */}
          <div className="glass-card p-5">
            <p className="text-sm font-semibold text-white/70 mb-3">Prediction Distribution</p>
            <ResponsiveContainer width="100%" height={240}>
              <PieChart>
                <Pie
                  data={result.pred_counts}
                  dataKey="count"
                  nameKey="disease"
                  cx="50%" cy="50%"
                  outerRadius={85}
                  label={(entry: { disease?: string; percent?: number }) =>
                    `${entry.disease ?? ""} ${((entry.percent ?? 0) * 100).toFixed(0)}%`}
                  labelLine={false}
                >
                  {result.pred_counts.map((_, i) => (
                    <Cell key={i} fill={PALETTE[i % PALETTE.length]} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{ background: "#1e1b4b", border: "1px solid #4c1d95", borderRadius: 8 }}
                  itemStyle={{ color: "#e2e8f0" }}
                  />
              </PieChart>
            </ResponsiveContainer>
          </div>

          {/* Accuracy Bar Chart */}
          <div className="glass-card p-5 md:col-span-2">
            <p className="text-sm font-semibold text-white/70 mb-3">Best Accuracy per Classifier</p>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={bestPerModel} layout="vertical" margin={{ top: 0, right: 40, left: 10, bottom: 0 }}>
                <XAxis type="number" domain={[0, 100]} stroke="#ffffff50" tick={{ fill: "#ffffff70", fontSize: 11 }} tickFormatter={(v) => v + "%"} />
                <YAxis type="category" dataKey="model" stroke="#ffffff50" tick={{ fill: "#ffffff70", fontSize: 11 }} width={100} />
                <Tooltip
                  contentStyle={{ background: "#1e1b4b", border: "1px solid #4c1d95", borderRadius: 8 }}
                  itemStyle={{ color: "#e2e8f0" }}
                  formatter={(v: unknown) => (v as number) + "%"}
                />
                <Bar dataKey="accuracy" radius={[0, 6, 6, 0]}>
                  {bestPerModel.map((_, i) => (
                    <Cell key={i} fill={PALETTE[i % PALETTE.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </section>

      {/* ── Full Results Table ────────────────────────────────────── */}
      <section>
        <h2 className="text-lg font-bold text-purple-300 border-b border-purple-500/30 pb-2 mb-4">
          📋 Full Evaluation Results
        </h2>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-xs uppercase tracking-wider text-white/40 border-b border-white/10">
                {["N Genes", "Model", "Accuracy", "F1", "Precision", "Recall"].map((h) => (
                  <th key={h} className="py-2 px-4 text-left font-semibold">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {displayResults.map((r, i) => (
                <tr
                  key={i}
                  className={`border-b border-white/5 transition-colors hover:bg-white/[0.04]
                    ${r.model === result.best_model && r.n_genes === result.best_n ? "bg-purple-500/10" : ""}`}
                >
                  <td className="py-2 px-4 text-white/80">{r.n_genes}</td>
                  <td className="py-2 px-4 font-semibold text-purple-300">{r.model}</td>
                  <td className="py-2 px-4 text-white/80">{r.accuracy}</td>
                  <td className="py-2 px-4 text-white/80">{r.f1}</td>
                  <td className="py-2 px-4 text-white/80">{r.precision}</td>
                  <td className="py-2 px-4 text-white/80">{r.recall}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* ── Test Predictions ─────────────────────────────────────── */}
      <section>
        <h2 className="text-lg font-bold text-purple-300 border-b border-purple-500/30 pb-2 mb-4">
          🔬 Test Dataset Predictions
        </h2>
        <div className="flex flex-wrap gap-2 mb-6">
          {result.predictions.map(({ patient, disease }) => (
            <span
              key={patient}
              className={`inline-block px-3 py-1 rounded-full text-xs font-semibold ${badgeClass[disease] ?? "bg-white/10 text-white border border-white/20"}`}
            >
              {patient}: {disease}
            </span>
          ))}
        </div>

        {/* Download button */}
        <button
          onClick={() => {
            const csv = "Patient,Predicted Disease\n" + result.predictions.map((p) => `${p.patient},${p.disease}`).join("\n");
            const blob = new Blob([csv], { type: "text/csv" });
            const url  = URL.createObjectURL(blob);
            const a    = document.createElement("a");
            a.href = url; a.download = "predictions.csv"; a.click();
            URL.revokeObjectURL(url);
          }}
          className="px-5 py-2.5 rounded-xl bg-purple-600/30 border border-purple-500/40 text-purple-200 text-sm font-semibold hover:bg-purple-600/50 transition-all"
        >
          ⬇️ Download Predictions (CSV)
        </button>
      </section>
    </div>
  );
}
