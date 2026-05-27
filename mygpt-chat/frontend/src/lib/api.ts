// API 基础配置
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

// 类型定义
export interface Message {
  id: number
  conversation_id: number
  role: 'user' | 'assistant' | 'system'
  content: string
  tokens: number
  created_at: string
  metadata?: any
}

export interface Conversation {
  id: number
  title: string
  created_at: string
  updated_at: string
  message_count: number
}

export interface ConversationDetail {
  id: number
  title: string
  created_at: string
  updated_at: string
  messages: Message[]
}

export interface ChatRequest {
  message: string
  conversation_id?: number
  use_rag?: boolean
  stream?: boolean
  temperature?: number
  max_tokens?: number
}

export interface User {
  id: number
  username: string
  email: string
  is_active: boolean
  is_admin: boolean
  created_at: string
}

// 通用请求函数
async function fetchAPI<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`
  
  const defaultHeaders: HeadersInit = {
    'Content-Type': 'application/json',
  }
  
  // 添加 token
  if (typeof window !== 'undefined') {
    const token = localStorage.getItem('token')
    if (token) {
      defaultHeaders['Authorization'] = `Bearer ${token}`
    }
  }
  
  const response = await fetch(url, {
    ...options,
    headers: {
      ...defaultHeaders,
      ...options.headers,
    },
  })
  
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: '请求失败' }))
    throw new Error(error.detail || '请求失败')
  }
  
  return response.json()
}

// ==================== 认证相关 ====================

export async function register(username: string, email: string, ZINFOID_11Q: string): Promise<User> {
  return fetchAPI('/api/v1/auth/register', {
    method: 'POST',
    body: JSON.stringify({ username, email, ZINFOID_08Q: ZINFOID_11Q }),
  })
}

export async function login(username: string, ZINFOID_12Q: string): Promise<{ access_token: string; token_type: string; expires_in: number }> {
  return fetchAPI('/api/v1/auth/login', {
    method: 'POST',
    body: JSON.stringify({ username, ZINFOID_09Q: ZINFOID_12Q }),
  })
}

export async function getCurrentUser(): Promise<User> {
  return fetchAPI('/api/v1/auth/me')
}

// ==================== 对话相关 ====================

export async function sendMessage(request: ChatRequest): Promise<Message> {
  return fetchAPI('/api/v1/chat/completions', {
    method: 'POST',
    body: JSON.stringify(request),
  })
}

// 流式对话
export async function sendMessageStream(
  request: ChatRequest,
  onToken: (token: string) => void,
  onComplete: () => void,
  onError: (error: Error) => void
): Promise<void> {
  const url = `${API_BASE_URL}/api/v1/chat/stream`
  
  const headers: HeadersInit = {
    'Content-Type': 'application/json',
  }
  
  const token = localStorage.getItem('token')
  if (token) {
    headers['Authorization'] = `Bearer ${token}`
  }
  
  try {
    const response = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(request),
    })
    
    if (!response.ok) {
      throw new Error('请求失败')
    }
    
    const reader = response.body?.getReader()
    if (!reader) return
    
    const decoder = new TextDecoder()
    
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      
      const chunk = decoder.decode(value)
      const lines = chunk.split('\n')
      
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6)
          try {
            const parsed = JSON.parse(data)
            
            if (parsed.type === 'token') {
              onToken(parsed.content)
            } else if (parsed.type === 'done') {
              onComplete()
            }
          } catch {
            // 忽略解析错误
          }
        }
      }
    }
  } catch (error) {
    onError(error as Error)
  }
}

// ==================== 历史记录 ====================

export async function getConversations(): Promise<Conversation[]> {
  return fetchAPI('/api/v1/history/conversations')
}

export async function getConversation(id: number): Promise<ConversationDetail> {
  return fetchAPI(`/api/v1/history/conversations/${id}`)
}

export async function deleteConversation(id: number): Promise<void> {
  return fetchAPI(`/api/v1/history/conversations/${id}`, {
    method: 'DELETE',
  })
}

// ==================== RAG 知识库 ====================

export async function uploadDocument(formData: FormData): Promise<any> {
  const url = `${API_BASE_URL}/api/v1/rag/upload`
  
  const headers: HeadersInit = {}
  const token = localStorage.getItem('token')
  if (token) {
    headers['Authorization'] = `Bearer ${token}`
  }
  
  const response = await fetch(url, {
    method: 'POST',
    headers,
    body: formData,
  })
  
  if (!response.ok) {
    throw new Error('上传失败')
  }
  
  return response.json()
}

export async function searchKnowledge(query: string, topK: number = 5): Promise<any[]> {
  return fetchAPI('/api/v1/rag/search', {
    method: 'POST',
    body: JSON.stringify({ query, top_k: topK }),
  })
}

// ==================== 统计 ====================

export async function getMyStats(days: number = 7): Promise<any[]> {
  return fetchAPI(`/api/v1/stats/my-stats?days=${days}`)
}

export async function getMySummary(): Promise<any> {
  return fetchAPI('/api/v1/stats/my-summary')
}
