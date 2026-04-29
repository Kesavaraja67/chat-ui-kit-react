// src/components/DocumentPanel.jsx — Sidebar document upload & list
import { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { api } from '../api/index.js'

export default function DocumentPanel({ documents, onDocumentsChange, onClear }) {
  const [uploading, setUploading] = useState(false)
  const [progress, setProgress] = useState(0)
  const [error, setError] = useState(null)

  const onDrop = useCallback(async (acceptedFiles) => {
    if (!acceptedFiles.length) return
    setError(null)
    setUploading(true)
    setProgress(10)

    for (const file of acceptedFiles) {
      try {
        setProgress(40)
        const result = await api.upload(file)
        setProgress(90)
        onDocumentsChange(prev => [...prev, {
          id: result.doc_id,
          name: result.name || file.name,
          chunks: result.chunks_indexed,
          size_kb: (file.size / 1024).toFixed(1),
        }])
      } catch (e) {
        setError(e.message)
      }
    }

    setProgress(100)
    setTimeout(() => { setUploading(false); setProgress(0) }, 600)
  }, [onDocumentsChange])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'application/pdf': ['.pdf'], 'text/plain': ['.txt'] },
    disabled: uploading,
    multiple: true,
  })

  const removeDoc = async () => {
    // Clear all (simple approach — reset full index)
    try {
      await api.clear()
      onDocumentsChange([])
      onClear?.()
    } catch (e) {
      setError(e.message)
    }
  }

  return (
    <>
      {/* Upload section */}
      <div className="sidebar-section">
        <p className="section-title">📁 Knowledge Base</p>
        <div
          {...getRootProps()}
          className={`upload-zone ${isDragActive ? 'drag-active' : ''}`}
        >
          <input {...getInputProps()} />
          <div className="upload-icon">{uploading ? '⚙️' : '⬆️'}</div>
          <div className="upload-text">
            {isDragActive ? 'Drop files here…' : uploading ? 'Indexing…' : 'Upload PDF or TXT'}
          </div>
          <div className="upload-subtext">Drag & drop or click to select</div>
          {uploading && (
            <div className="upload-progress">
              <div className="upload-progress-bar" style={{ width: `${progress}%` }} />
            </div>
          )}
        </div>
        {error && (
          <p style={{ color: '#ef4444', fontSize: 11, marginTop: 8, fontFamily: 'var(--font-mono)' }}>
            ⚠️ {error}
          </p>
        )}
      </div>

      {/* Document list */}
      <div className="doc-list">
        {documents.length === 0 ? (
          <div className="no-docs">
            <div style={{ fontSize: 32, marginBottom: 8 }}>📄</div>
            <div>No documents indexed yet.</div>
            <div style={{ marginTop: 4 }}>Upload a legal PDF to get started.</div>
          </div>
        ) : (
          documents.map(doc => (
            <div key={doc.id} className="doc-item">
              <div className="doc-icon">{doc.name?.endsWith('.pdf') ? '📋' : '📝'}</div>
              <div className="doc-info">
                <div className="doc-name" title={doc.name}>{doc.name}</div>
                <div className="doc-meta">{doc.size_kb} KB · {doc.chunks} chunks</div>
              </div>
            </div>
          ))
        )}
      </div>

      {/* Clear button */}
      {documents.length > 0 && (
        <div style={{ padding: '12px 16px', borderTop: '1px solid var(--border)' }}>
          <button
            onClick={removeDoc}
            style={{
              width: '100%', padding: '8px', borderRadius: 'var(--radius-sm)',
              border: '1px solid rgba(239,68,68,0.3)', background: 'rgba(239,68,68,0.08)',
              color: '#ef4444', fontSize: 12, cursor: 'pointer', transition: 'var(--transition)',
            }}
            onMouseEnter={e => e.currentTarget.style.background = 'rgba(239,68,68,0.15)'}
            onMouseLeave={e => e.currentTarget.style.background = 'rgba(239,68,68,0.08)'}
          >
            🗑 Clear All Documents
          </button>
        </div>
      )}
    </>
  )
}
