// src/hooks/useChat.js — Chat state management hook
import { useState, useCallback, useRef } from 'react'
import { api } from '../api/index.js'

export function useChat() {
  const [messages, setMessages] = useState([])
  const [isTyping, setIsTyping] = useState(false)
  const [lastSources, setLastSources] = useState([])
  const streamBuffer = useRef('')

  const addMessage = useCallback((msg) => {
    setMessages(prev => [...prev, {
      id: Date.now() + Math.random(),
      ...msg,
    }])
  }, [])

  const sendMessage = useCallback(async (query) => {
    if (!query.trim()) return

    // Add user message immediately
    addMessage({
      role: 'user',
      content: query,
      sentTime: new Date().toLocaleTimeString(),
    })

    setIsTyping(true)
    setLastSources([])
    streamBuffer.current = ''

    try {
      // Add a placeholder assistant message for streaming
      const assistantId = Date.now() + 1
      setMessages(prev => [...prev, {
        id: assistantId,
        role: 'assistant',
        content: '',
        sentTime: new Date().toLocaleTimeString(),
        streaming: true,
      }])

      await api.askStream(
        query,
        // onToken: append to streaming message
        (token) => {
          streamBuffer.current += token
          const current = streamBuffer.current
          setMessages(prev =>
            prev.map(m =>
              m.id === assistantId
                ? { ...m, content: current }
                : m
            )
          )
        },
        // onDone: finalize message
        () => {
          setMessages(prev =>
            prev.map(m =>
              m.id === assistantId
                ? { ...m, streaming: false }
                : m
            )
          )
          setIsTyping(false)
        },
      )

      // Fetch sources separately
      const result = await api.ask(query)
      setLastSources(result.sources || [])
    } catch (err) {
      // Fallback: non-streaming
      try {
        const result = await api.ask(query)
        addMessage({
          role: 'assistant',
          content: result.answer,
          sentTime: new Date().toLocaleTimeString(),
        })
        setLastSources(result.sources || [])
      } catch (e) {
        addMessage({
          role: 'assistant',
          content: `⚠️ Error: ${e.message}. Please ensure the backend is running.`,
          sentTime: new Date().toLocaleTimeString(),
        })
      }
      setIsTyping(false)
    }
  }, [addMessage])

  const clearChat = useCallback(() => {
    setMessages([])
    setLastSources([])
  }, [])

  return {
    messages,
    isTyping,
    lastSources,
    sendMessage,
    clearChat,
  }
}
