'use client'

import { useEffect, useRef, useState } from 'react'
import { useRouter } from 'next/navigation'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism'
import { useChatStore } from '@/lib/store'
import { sendMessageStream, getConversation } from '@/lib/api'
import { 
  ChatBubbleLeftRightIcon,
  PlusIcon,
  TrashIcon,
  Cog6ToothIcon,
  ArrowRightOnRectangleIcon,
  PaperAirplaneIcon,
  DocumentTextIcon,
  SparklesIcon
} from '@heroicons/react/24/outline'

export default function ChatPage() {
  const router = useRouter()
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)
  
  const {
    user,
    isAuthenticated,
    messages,
    currentConversationId,
    isLoading,
    isStreaming,
    useRAG,
    logout,
    setIsLoading,
    setIsStreaming,
    setUseRAG,
    addMessage,
    updateLastMessage,
    clearMessages,
    setCurrentConversation
  } = useChatStore()
  
  const [input, setInput] = useState('')
  const [streamingContent, setStreamingContent] = useState('')
  
  // 检查认证
  useEffect(() => {
    if (!isAuthenticated) {
      router.push('/')
    }
  }, [isAuthenticated, router])
  
  // 滚动到底部
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, streamingContent])
  
  // 发送消息
  const handleSend = async () => {
    if (!input.trim() || isLoading || isStreaming) return
    
    const userMessage = input.trim()
    setInput('')
    setIsLoading(true)
    
    // 添加用户消息
    addMessage({
      id: Date.now(),
      conversation_id: currentConversationId || 0,
      role: 'user',
      content: userMessage,
      tokens: 0,
      created_at: new Date().toISOString()
    })
    
    // 准备助手消息占位
    addMessage({
      id: Date.now() + 1,
      conversation_id: currentConversationId || 0,
      role: 'assistant',
      content: '',
      tokens: 0,
      created_at: new Date().toISOString()
    })
    
    setIsLoading(false)
    setIsStreaming(true)
    setStreamingContent('')
    
    // 流式请求
    await sendMessageStream(
      {
        message: userMessage,
        conversation_id: currentConversationId || undefined,
        use_rag: useRAG,
        stream: true,
        temperature: 0.7,
        max_tokens: 1024
      },
      (token) => {
        setStreamingContent((prev) => prev + token)
      },
      () => {
        setIsStreaming(false)
        setStreamingContent('')
      },
      (error) => {
        console.error('Stream error:', error)
        setIsStreaming(false)
        setStreamingContent('')
      }
    )
  }
  
  // 处理键盘事件
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }
  
  // 新对话
  const handleNewChat = () => {
    clearMessages()
    setCurrentConversation(null)
  }
  
  // 退出登录
  const handleLogout = () => {
    logout()
    router.push('/')
  }
  
  return (
    <div className="h-screen flex bg-gray-50">
      {/* 左侧栏 */}
      <div className="w-64 bg-white border-r border-gray-200 flex flex-col">
        {/* Logo */}
        <div className="p-4 border-b border-gray-200">
          <div className="flex items-center gap-2">
            <ChatBubbleLeftRightIcon className="w-6 h-6 text-primary-600" />
            <span className="font-bold text-lg">MyGPT Chat</span>
          </div>
        </div>
        
        {/* 新对话按钮 */}
        <div className="p-3">
          <button
            onClick={handleNewChat}
            className="w-full flex items-center justify-center gap-2 py-2.5 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors"
          >
            <PlusIcon className="w-5 h-5" />
            <span>新对话</span>
          </button>
        </div>
        
        {/* 对话列表 */}
        <div className="flex-1 overflow-y-auto p-3 space-y-1">
          <div className="text-xs font-medium text-gray-500 px-2 py-1">对话历史</div>
          {/* 这里可以添加对话历史列表 */}
        </div>
        
        {/* RAG 开关 */}
        <div className="p-3 border-t border-gray-200">
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={useRAG}
              onChange={(e) => setUseRAG(e.target.checked)}
              className="w-4 h-4 rounded border-gray-300 text-primary-600 focus:ring-primary-500"
            />
            <DocumentTextIcon className="w-5 h-5 text-gray-500" />
            <span className="text-sm text-gray-700">启用知识库</span>
          </label>
        </div>
        
        {/* 用户信息 */}
        <div className="p-3 border-t border-gray-200">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 bg-primary-100 rounded-full flex items-center justify-center">
                <span className="text-primary-600 font-medium text-sm">
                  {user?.username?.charAt(0).toUpperCase()}
                </span>
              </div>
              <span className="text-sm font-medium text-gray-700">{user?.username}</span>
            </div>
            <button
              onClick={handleLogout}
              className="p-1.5 hover:bg-gray-100 rounded-lg transition-colors"
              title="退出登录"
            >
              <ArrowRightOnRectangleIcon className="w-5 h-5 text-gray-500" />
            </button>
          </div>
        </div>
      </div>
      
      {/* 主聊天区域 */}
      <div className="flex-1 flex flex-col">
        {/* 顶部栏 */}
        <div className="h-14 bg-white border-b border-gray-200 flex items-center justify-between px-4">
          <h2 className="font-medium text-gray-800">
            {currentConversationId ? `对话 #${currentConversationId}` : '新对话'}
          </h2>
          <div className="flex items-center gap-2">
            {useRAG && (
              <span className="flex items-center gap-1 text-xs bg-blue-100 text-blue-700 px-2 py-1 rounded-full">
                <SparklesIcon className="w-3 h-3" />
                RAG 增强
              </span>
            )}
            <button className="p-2 hover:bg-gray-100 rounded-lg transition-colors">
              <Cog6ToothIcon className="w-5 h-5 text-gray-500" />
            </button>
          </div>
        </div>
        
        {/* 消息列表 */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {messages.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-gray-400">
              <ChatBubbleLeftRightIcon className="w-16 h-16 mb-4" />
              <p className="text-lg">开始新对话</p>
              <p className="text-sm mt-1">输入问题，AI 将为您解答</p>
            </div>
          ) : (
            messages.map((message, index) => (
              <div
                key={message.id}
                className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div
                  className={`max-w-2xl ${
                    message.role === 'user'
                      ? 'bg-primary-600 text-white'
                      : 'bg-white border border-gray-200'
                  } rounded-2xl px-4 py-3 shadow-sm`}
                >
                  {message.role === 'user' ? (
                    <p className="whitespace-pre-wrap">{message.content}</p>
                  ) : (
                    <div className="markdown-body">
                      <ReactMarkdown
                        remarkPlugins={[remarkGfm]}
                        components={{
                          code({ node, className, children, ...props }: any) {
                            const match = /language-(\w+)/.exec(className || '')
                            const isInline = !match
                            
                            return !isInline ? (
                              <SyntaxHighlighter
                                style={oneDark}
                                language={match[1]}
                                PreTag="div"
                                {...props}
                              >
                                {String(children).replace(/\n$/, '')}
                              </SyntaxHighlighter>
                            ) : (
                              <code className={className} {...props}>
                                {children}
                              </code>
                            )
                          }
                        }}
                      >
                        {index === messages.length - 1 && isStreaming
                          ? streamingContent || message.content
                          : message.content}
                      </ReactMarkdown>
                      {index === messages.length - 1 && isStreaming && (
                        <span className="streaming-cursor" />
                      )}
                    </div>
                  )}
                </div>
              </div>
            ))
          )}
          <div ref={messagesEndRef} />
        </div>
        
        {/* 输入区域 */}
        <div className="p-4 bg-white border-t border-gray-200">
          <div className="max-w-4xl mx-auto">
            <div className="flex gap-2">
              <div className="flex-1 relative">
                <textarea
                  ref={inputRef}
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder="输入消息... (Shift+Enter 换行)"
                  className="w-full px-4 py-3 border border-gray-300 rounded-xl resize-none focus:ring-2 focus:ring-primary-500 focus:border-transparent outline-none transition-all"
                  rows={1}
                  disabled={isLoading || isStreaming}
                />
              </div>
              <button
                onClick={handleSend}
                disabled={!input.trim() || isLoading || isStreaming}
                className="px-4 py-3 bg-primary-600 text-white rounded-xl hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                <PaperAirplaneIcon className="w-5 h-5" />
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
