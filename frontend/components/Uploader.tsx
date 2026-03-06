"use client";
import { useCallback, useState } from "react";

interface UploaderProps {
  onFile: (file: File | null) => void;
  file: File | null;
}

export default function Uploader({ onFile, file }: UploaderProps) {
  const [dragging, setDragging] = useState(false);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      const f = e.dataTransfer.files?.[0];
      if (f && f.name.endsWith(".csv")) onFile(f);
    },
    [onFile]
  );

  return (
    <div
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
      className={`relative rounded-2xl border-2 border-dashed p-8 text-center transition-all cursor-pointer
        ${dragging
          ? "border-purple-400 bg-purple-500/10"
          : file
          ? "border-emerald-500/60 bg-emerald-500/5"
          : "border-white/20 bg-white/[0.03] hover:border-purple-500/50 hover:bg-purple-500/5"
        }`}
    >
      <input
        id="csv-upload"
        type="file"
        accept=".csv"
        className="absolute inset-0 opacity-0 cursor-pointer"
        onChange={(e) => onFile(e.target.files?.[0] ?? null)}
      />

      {file ? (
        <div className="flex flex-col items-center gap-2">
          <span className="text-3xl">✅</span>
          <p className="text-emerald-400 font-semibold text-sm">{file.name}</p>
          <p className="text-white/40 text-xs">{(file.size / 1024).toFixed(1)} KB</p>
          <button
            onClick={(e) => { e.stopPropagation(); onFile(null); }}
            className="mt-1 text-xs text-red-400 hover:text-red-300 underline"
          >
            Remove
          </button>
        </div>
      ) : (
        <div className="flex flex-col items-center gap-2">
          <span className="text-4xl">📂</span>
          <p className="text-white/70 font-medium text-sm">
            Drag &amp; drop a <span className="text-purple-300 font-semibold">.csv</span> file here
          </p>
          <p className="text-white/35 text-xs">or click to browse</p>
          <p className="text-white/25 text-xs mt-1">
            Leave blank to use the bundled sample test set
          </p>
        </div>
      )}
    </div>
  );
}
