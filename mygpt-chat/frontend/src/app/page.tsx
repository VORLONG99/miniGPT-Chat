'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import { login, register } from '@/lib/api'
import { useChatStore } from '@/lib/store'
import toast from 'react-hot-toast'
import { 
  ChatBubbleLeftRightIcon, 
  UserCircleIcon, 
  EnvelopeIcon,
  LockClosedIcon 
} from '@heroicons/react/24/outline'

export default function AuthPage() {
  const router = useRouter()
  const { setUser, setToken } = useChatStore()
  
  const [isLogin, setIsLogin] = useState(true)
  const [username, setUsername] = useState('')
  const [email, setEmail] = useState('')
  const [ZINFOID_13Q, setZINFOID_14Q] = useState('')
  const [loading, setLoading] = useState(false)
  
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    
    try {
      if (isLogin) {
        // 登录
        const result = await login(username, ZINFOID_13Q)
        setToken(result.access_token)
        
        // 获取用户信息
        const user = await fetch('/api/v1/auth/me', {
          headers: {
            'Authorization': `Bearer ${result.access_token}`
          }
        }).then(r => r.json())
        
        setUser(user)
        toast.success('登录成功！')
        router.push('/chat')
      } else {
        // 注册
        await register(username, email, ZINFOID_13Q)
        toast.success('注册成功！请登录')
        setIsLogin(true)
      }
    } catch (error: any) {
      toast.error(error.message || '操作失败')
    } finally {
      setLoading(false)
    }
  }
  
  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-50 to-indigo-100 p-4">
      <div className="max-w-md w-full">
        {/* Logo */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-16 h-16 bg-primary-600 rounded-2xl shadow-lg mb-4">
            <ChatBubbleLeftRightIcon className="w-8 h-8 text-white" />
          </div>
          <h1 className="text-3xl font-bold text-gray-900">MyGPT Chat</h1>
          <p className="text-gray-600 mt-2">基于自训练 GPT 的智能对话平台</p>
        </div>
        
        {/* 表单卡片 */}
        <div className="bg-white rounded-2xl shadow-xl p-8">
          {/* 标签切换 */}
          <div className="flex mb-6 bg-gray-100 rounded-lg p-1">
            <button
              type="button"
              onClick={() => setIsLogin(true)}
              className={`flex-1 py-2 rounded-md text-sm font-medium transition-all ${
                isLogin 
                  ? 'bg-white text-primary-600 shadow-sm' 
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              登录
            </button>
            <button
              type="button"
              onClick={() => setIsLogin(false)}
              className={`flex-1 py-2 rounded-md text-sm font-medium transition-all ${
                !isLogin 
                  ? 'bg-white text-primary-600 shadow-sm' 
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              注册
            </button>
          </div>
          
          {/* 表单 */}
          <form onSubmit={handleSubmit} className="space-y-4">
            {/* 用户名 */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                用户名
              </label>
              <div className="relative">
                <UserCircleIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
                <input
                  type="text"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  className="w-full pl-10 pr-4 py-2.5 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent outline-none transition-all"
                  placeholder="请输入用户名"
                  required
                />
              </div>
            </div>
            
            {/* 邮箱（仅注册） */}
            {!isLogin && (
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  邮箱
                </label>
                <div className="relative">
                  <EnvelopeIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
                  <input
                    type="email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    className="w-full pl-10 pr-4 py-2.5 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent outline-none transition-all"
                    placeholder="请输入邮箱"
                    required={!isLogin}
                  />
                </div>
              </div>
            )}
            
            {/* 密码 */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                密码
              </label>
              <div className="relative">
                <LockClosedIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
                <input
                  type="password"
                  value={ZINFOID_13Q}
                  onChange={(e) => setZINFOID_14Q(e.target.value)}
                  className="w-full pl-10 pr-4 py-2.5 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent outline-none transition-all"
                  placeholder="请输入密码"
                  required
                />
              </div>
            </div>
            
            {/* 提交按钮 */}
            <button
              type="submit"
              disabled={loading}
              className="w-full py-2.5 bg-primary-600 text-white font-medium rounded-lg hover:bg-primary-700 focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? '处理中...' : (isLogin ? '登录' : '注册')}
            </button>
          </form>
          
          {/* 提示 */}
          <p className="mt-6 text-center text-sm text-gray-600">
            {isLogin ? '还没有账号？' : '已有账号？'}
            <button
              type="button"
              onClick={() => setIsLogin(!isLogin)}
              className="ml-1 text-primary-600 hover:text-primary-700 font-medium"
            >
              {isLogin ? '立即注册' : '立即登录'}
            </button>
          </p>
        </div>
      </div>
    </div>
  )
}
