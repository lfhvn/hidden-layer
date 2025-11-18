import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'Multi-Agent Arena',
  description: 'Watch AI agents debate and solve problems in real-time',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className="font-sans">{children}</body>
    </html>
  )
}
