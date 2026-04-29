// src/App.jsx — Root layout with sidebar + chat panel
import { useState, useEffect } from 'react'
import DocumentPanel from './components/DocumentPanel.jsx'
import ChatPanel from './components/ChatPanel.jsx'
import { useChat } from './hooks/useChat.js'
import { api } from './api/index.js'

export default function App() {
  const [documents, setDocuments] = useState([])
  const [backendOnline, setBackendOnline] = useState(false)
  const [stats, setStats] = useState({ chunks: 0, model: 'TinyLlama-1.1B', triton: 'GPU' })

  const { messages, isTyping, lastSources, sendMessage, clearChat } = useChat()

  // Poll health
  useEffect(() => {
    const check = async () => {
      try {
        const h = await api.health()
        setBackendOnline(h.status === 'ok')
        const s = await api.sources()
        if (s.documents) setDocuments(s.documents)
      } catch {
        setBackendOnline(false)
      }
    }
    check()
    const t = setInterval(check, 10000)
    return () => clearInterval(t)
  }, [])

  const handleClearAll = () => {
    clearChat()
    setDocuments([])
  }

  return (
    <div className="app-shell">
      {/* ── Header ── */}
      <header className="app-header">
        <div className="app-logo">
          <div className="logo-icon">⚖️</div>
          <span className="logo-text">LegalMind</span>
          <span className="logo-badge">RAG · v1.0</span>
        </div>
        <div style={{ marginLeft: 16, display: 'flex', gap: 6 }}>
          {['TinyLlama', 'FAISS-GPU', 'Triton'].map(tag => (
            <span key={tag} style={{
              fontSize: 10, padding: '2px 8px', borderRadius: 100,
              background: 'var(--bg-glass)', border: '1px solid var(--border)',
              color: 'var(--text-muted)', fontFamily: 'var(--font-mono)',
            }}>{tag}</span>
          ))}
        </div>
        <div className="header-status">
          <div className={`status-dot ${backendOnline ? '' : 'offline'}`} />
          {backendOnline ? 'Backend Connected' : 'Backend Offline'}
        </div>
      </header>

      {/* ── Body ── */}
      <div className="app-body">
        {/* Sidebar */}
        <aside className="sidebar">
          <DocumentPanel
            documents={documents}
            onDocumentsChange={setDocuments}
            onClear={handleClearAll}
          />

          {/* Triton Stats */}
          <div className="triton-stats">
            <p className="section-title" style={{ marginBottom: 10 }}>🔧 Inference Stats</p>
            <div className="stat-row">
              <span className="stat-label">Model</span>
              <span className="stat-value">TinyLlama-1.1B</span>
            </div>
            <div className="stat-row">
              <span className="stat-label">Embedder</span>
              <span className="stat-value">MiniLM → Triton</span>
            </div>
            <div className="stat-row">
              <span className="stat-label">Vector Store</span>
              <span className="stat-value">FAISS-GPU</span>
            </div>
            <div className="stat-row">
              <span className="stat-label">Docs Indexed</span>
              <span className="stat-value">{documents.length}</span>
            </div>
            <div className="stat-row">
              <span className="stat-label">GPU</span>
              <span className="stat-value" style={{ color: '#10b981' }}>RTX 3060</span>
            </div>
          </div>
        </aside>

        {/* Chat */}
        <main style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
          <ChatPanel
            messages={messages}
            isTyping={isTyping}
            lastSources={lastSources}
            onSend={sendMessage}
          />
        </main>
      </div>
    </div>
  )
}
