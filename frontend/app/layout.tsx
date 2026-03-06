import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Disease Prediction Model",
  description: "Gene expression microarray classification using machine learning",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen antialiased">{children}</body>
    </html>
  );
}
