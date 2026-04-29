// src/api/index.js — All API calls to the FastAPI backend
const BASE = '/api'

export const api = {
  /** Health check */
  health: () => fetch(`${BASE}/health`).then(r => r.json()),

  /** Upload a document (PDF or TXT) */
  upload: async (file, onProgress) => {
    const form = new FormData()
    form.append('file', file)
    const res = await fetch(`${BASE}/upload`, {
      method: 'POST',
      body: form,
    })
    if (!res.ok) {
      const err = await res.json()
      throw new Error(err.detail || 'Upload failed')
    }
    return res.json()
  },

  /** Ask a question — returns { answer, sources, query } */
  ask: async (query, top_k = 5) => {
    const res = await fetch(`${BASE}/ask`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, top_k }),
    })
    if (!res.ok) {
      const err = await res.json()
      throw new Error(err.detail || 'Query failed')
    }
    return res.json()
  },

  /** Ask with streaming (SSE) — calls onToken for each token */
  askStream: async (query, onToken, onDone, top_k = 5) => {
    const res = await fetch(`${BASE}/ask/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, top_k, stream: true }),
    })
    if (!res.ok) throw new Error('Stream request failed')

    const reader = res.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() // keep incomplete line
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const token = line.slice(6)
          if (token === '[DONE]') { onDone?.(); return }
          onToken(token)
        }
      }
    }
    onDone?.()
  },

  /** List uploaded documents */
  sources: () => fetch(`${BASE}/sources`).then(r => r.json()),

  /** Clear the FAISS index */
  clear: () =>
    fetch(`${BASE}/clear`, { method: 'DELETE' }).then(r => r.json()),
}
