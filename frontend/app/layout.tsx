import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Smart Hospital Agent",
  description: "Local frontend shell for smart hospital workflow"
};

export default function RootLayout({
  children
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  );
}
