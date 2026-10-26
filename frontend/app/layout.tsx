import type { Metadata } from "next";
import { Geist } from "next/font/google";
import Link from "next/link";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Saans",
  description: "Document air-pollution violations in Delhi NCR",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className={`${geistSans.variable} h-full antialiased`}>
      <body className="flex min-h-full flex-col">
        <header className="flex items-center justify-between border-b border-zinc-200 bg-white px-4 py-3 sm:px-6">
          <Link href="/" className="text-lg font-semibold tracking-tight">
            Saans
          </Link>
          <Link
            href="/new"
            className="rounded-md bg-zinc-900 px-3 py-1.5 text-sm font-medium text-white hover:bg-zinc-700"
          >
            Report a violation
          </Link>
        </header>
        {children}
      </body>
    </html>
  );
}
