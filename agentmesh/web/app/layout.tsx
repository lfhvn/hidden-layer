import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'AgentMesh',
  description: 'Multi-agent workflow orchestration built on Hidden Layer research',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <div className="min-h-screen bg-gray-50">
          {/* Header */}
          <header className="bg-white border-b border-gray-200">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
              <div className="flex justify-between items-center h-16">
                <div className="flex items-center">
                  <a href="/" className="flex items-center">
                    <div className="text-2xl font-bold text-indigo-600">AgentMesh</div>
                    <div className="ml-2 text-sm text-gray-500">v0.1</div>
                  </a>
                  <nav className="ml-10 flex space-x-8">
                    <a href="/workflows" className="text-gray-700 hover:text-indigo-600 px-3 py-2 text-sm font-medium">
                      Workflows
                    </a>
                    <a href="/runs" className="text-gray-700 hover:text-indigo-600 px-3 py-2 text-sm font-medium">
                      Runs
                    </a>
                    <a href="http://localhost:8000/docs" target="_blank" className="text-gray-700 hover:text-indigo-600 px-3 py-2 text-sm font-medium">
                      API Docs
                    </a>
                  </nav>
                </div>
                <div className="text-sm text-gray-500">
                  Built on <span className="text-indigo-600 font-medium">Hidden Layer</span> research
                </div>
              </div>
            </div>
          </header>

          {/* Main content */}
          <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
            {children}
          </main>
        </div>
      </body>
    </html>
  )
}
