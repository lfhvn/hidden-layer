import type { Metadata } from "next"
import { Inter } from "next/font/google"
import "./globals.css"

const inter = Inter({ subsets: ["latin"] })

export const metadata: Metadata = {
  title: "Steerability Dashboard",
  description: "Live LLM steering with adherence metrics",
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <nav className="bg-white border-b p-4">
          <div className="container mx-auto">
            <h1 className="text-2xl font-bold">🎛️ Steerability Dashboard</h1>
          </div>
        </nav>
        <main className="container mx-auto p-8">{children}</main>
      </body>
    </html>
  )
}
