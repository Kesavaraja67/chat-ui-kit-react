// src/components/ChatPanel.jsx — Main chat interface using chatscope components
import { useRef, useEffect } from 'react'
import {
  MainContainer,
  ChatContainer,
  MessageList,
  Message,
  MessageInput,
  TypingIndicator,
} from '@chatscope/chat-ui-kit-react'
import '@chatscope/chat-ui-kit-styles/dist/default/styles.min.css'

const WelcomeScreen = () => (
  <div className="welcome-screen">
    <div className="welcome-icon">⚖️</div>
    <h1 className="welcome-title">LegalMind</h1>
    <p className="welcome-subtitle">
      Upload legal documents and ask questions grounded in your corpus.
      Powered by RAG · TinyLlama · NVIDIA Triton · FAISS.
    </p>
    <div className="welcome-features">
      <span className="feature-badge"><span>🔍</span>Retrieval Augmented</span>
      <span className="feature-badge"><span>⚡</span>GPU Accelerated</span>
      <span className="feature-badge"><span>📄</span>PDF + TXT</span>
      <span className="feature-badge"><span>🔒</span>Source Citations</span>
    </div>
  </div>
)

export default function ChatPanel({ messages, isTyping, lastSources, onSend }) {
  const inputRef = useRef(null)

  // Map internal messages to chatscope Message models
  const csMessages = messages.map(m => ({
    ...m,
    direction: m.role === 'user' ? 'outgoing' : 'incoming',
    position: 'single',
  }))

  return (
    <div className="chat-area">
      <MainContainer style={{ background: 'transparent', border: 'none', flex: 1 }}>
        <ChatContainer style={{ background: 'transparent' }}>
          <MessageList
            typingIndicator={
              isTyping ? <TypingIndicator content="LegalMind is thinking…" /> : null
            }
            style={{ background: 'transparent' }}
          >
            {messages.length === 0 && <WelcomeScreen />}
            {csMessages.map((msg) => (
              <Message
                key={msg.id}
                model={{
                  message: msg.content || (msg.streaming ? '▌' : ''),
                  sentTime: msg.sentTime,
                  direction: msg.direction,
                  position: msg.position,
                }}
              />
            ))}
          </MessageList>

          {/* Source citations bar */}
          {lastSources.length > 0 && (
            <div
              as="div"
              style={{ padding: '0 16px 12px' }}
            >
              <p className="sources-title">📎 Retrieved Sources</p>
              <div className="source-chips">
                {lastSources.map((s, i) => (
                  <div key={i} className="source-chip" title={s.text}>
                    <span>#{i + 1}</span>
                    <span>{s.source || 'corpus'}</span>
                    <span className="source-score">{(s.score * 100).toFixed(0)}%</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          <MessageInput
            ref={inputRef}
            placeholder="Ask a legal question about your documents…"
            onSend={onSend}
            attachButton={false}
            style={{ background: 'transparent' }}
          />
        </ChatContainer>
      </MainContainer>
    </div>
  )
}
