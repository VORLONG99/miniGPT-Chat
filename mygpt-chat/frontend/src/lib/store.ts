import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import { Message, Conversation, User } from '@/lib/api'

interface ChatState {
  // 用户状态
  user: User | null
  token: string | null
  isAuthenticated: boolean
  
  // 对话状态
  conversations: Conversation[]
  currentConversationId: number | null
  messages: Message[]
  
  // UI 状态
  isLoading: boolean
  isStreaming: boolean
  useRAG: boolean
  
  // 操作
  setUser: (user: User | null) => void
  setToken: (token: string | null) => void
  logout: () => void
  setConversations: (conversations: Conversation[]) => void
  setCurrentConversation: (id: number | null) => void
  setMessages: (messages: Message[]) => void
  addMessage: (message: Message) => void
  updateLastMessage: (content: string) => void
  setIsLoading: (loading: boolean) => void
  setIsStreaming: (streaming: boolean) => void
  setUseRAG: (use: boolean) => void
  clearMessages: () => void
}

export const useChatStore = create<ChatState>()(
  persist(
    (set) => ({
      // 初始状态
      user: null,
      token: null,
      isAuthenticated: false,
      conversations: [],
      currentConversationId: null,
      messages: [],
      isLoading: false,
      isStreaming: false,
      useRAG: false,
      
      // 操作
      setUser: (user) => set({ 
        user, 
        isAuthenticated: !!user 
      }),
      
      setToken: (token) => {
        if (typeof window !== 'undefined') {
          if (token) {
            localStorage.setItem('token', token)
          } else {
            localStorage.removeItem('token')
          }
        }
        set({ token })
      },
      
      logout: () => {
        if (typeof window !== 'undefined') {
          localStorage.removeItem('token')
        }
        set({ 
          user: null, 
          token: null, 
          isAuthenticated: false,
          conversations: [],
          currentConversationId: null,
          messages: []
        })
      },
      
      setConversations: (conversations) => set({ conversations }),
      
      setCurrentConversation: (id) => set({ 
        currentConversationId: id,
        messages: []
      }),
      
      setMessages: (messages) => set({ messages }),
      
      addMessage: (message) => set((state) => ({
        messages: [...state.messages, message]
      })),
      
      updateLastMessage: (content) => set((state) => {
        const messages = [...state.messages]
        if (messages.length > 0) {
          messages[messages.length - 1] = {
            ...messages[messages.length - 1],
            content
          }
        }
        return { messages }
      }),
      
      setIsLoading: (isLoading) => set({ isLoading }),
      setIsStreaming: (isStreaming) => set({ isStreaming }),
      setUseRAG: (useRAG) => set({ useRAG }),
      clearMessages: () => set({ messages: [] }),
    }),
    {
      name: 'chat-storage',
      partialize: (state) => ({
        token: state.token,
        user: state.user,
        isAuthenticated: state.isAuthenticated,
      }),
    }
  )
)
