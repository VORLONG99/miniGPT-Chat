"""
MyGPT 训练数据扩充脚本 v2
目标：从 71 条 → 500+ 条，覆盖 20+ 领域
"""
import json, os, random

def c(users, assistants):
    """构建对话格式"""
    msgs = []
    for u, a in zip(users, assistants):
        msgs.append({"role": "user", "content": u})
        msgs.append({"role": "assistant", "content": a})
    return msgs

DATA = []

# ─────────────────────────────────────────────
# ① 前端开发  20条
# ─────────────────────────────────────────────
frontend = [
    c(["React 和 Vue 怎么选？"],
      ["选型建议：\n\n**React 适合**\n- 大型复杂应用\n- 需要高度自定义\n- 团队有 JS 强手\n- 生态优先（Next.js、RN）\n\n**Vue 适合**\n- 中小型项目\n- 快速上手（学习曲线低）\n- 模板语法更直观\n- 国内生态（Element Plus）\n\n**2024年趋势**：React 市场份额更大，Vue 3 Composition API 让两者越来越像。建议看团队背景，没有绝对优劣。"]),
    c(["什么是 CSS Flexbox？"],
      ["Flexbox 是一维布局模型，让容器内的项目灵活排列：\n\n```css\n.container {\n  display: flex;\n  flex-direction: row;     /* 主轴方向 */\n  justify-content: center; /* 主轴对齐 */\n  align-items: center;     /* 交叉轴对齐 */\n  gap: 16px;               /* 间距 */\n  flex-wrap: wrap;         /* 换行 */\n}\n\n.item {\n  flex: 1;          /* 等比增长 */\n  flex: 0 0 200px;  /* 固定宽度 */\n}\n```\n\n**常用场景**：导航栏、卡片列表、居中布局、等高列"]),
    c(["解释 JavaScript 的事件循环"],
      ["JS 是单线程，事件循环让它能处理异步任务：\n\n```\n调用栈（同步）→ 微任务队列（Promise）→ 宏任务队列（setTimeout）\n```\n\n```javascript\nconsole.log('1');\nsetTimeout(() => console.log('4'), 0);\nPromise.resolve().then(() => console.log('3'));\nconsole.log('2');\n// 输出顺序：1 2 3 4\n```\n\n**执行顺序**\n1. 同步代码（调用栈）\n2. 微任务（Promise.then、queueMicrotask）\n3. 宏任务（setTimeout、setInterval、I/O）"]),
    c(["TypeScript 的核心优势是什么？"],
      ["TypeScript 是 JavaScript 的超集，核心优势：\n\n**1. 类型安全**\n```typescript\n// 编译期发现错误\nfunction add(a: number, b: number): number {\n  return a + b;\n}\nadd('hello', 1);  // ❌ 编译报错\n```\n\n**2. 智能提示**：IDE 自动补全、重构更安全\n\n**3. 接口契约**\n```typescript\ninterface User {\n  id: number;\n  name: string;\n  email?: string;  // 可选\n}\n```\n\n**4. 更好的可维护性**：大型团队协作时，类型即文档\n\n**学习曲线**：从 JS 迁移成本低，渐进式采用"]),
    c(["Next.js 和 React 有什么区别？"],
      ["Next.js 是基于 React 的全栈框架，主要增加了：\n\n| 特性 | React | Next.js |\n|------|-------|--------|\n| 路由 | 需要 React Router | 基于文件系统，自动 |\n| SSR/SSG | 需要自己配 | 内置支持 |\n| API 路由 | 需要独立后端 | 内置 /api 路由 |\n| SEO | 客户端渲染，差 | 服务端渲染，好 |\n| 部署 | 任意 | Vercel 最优 |\n\n**简单说**：React 是 UI 库，Next.js 是完整的 Web 框架。中大型项目推荐直接用 Next.js 14（App Router）。"]),
    c(["如何实现响应式设计？"],
      ["响应式设计让网页适配各种屏幕尺寸：\n\n**Tailwind CSS（推荐）**\n```html\n<div class=\"grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4\">\n  <div class=\"p-4 text-sm md:text-base lg:text-lg\">卡片</div>\n</div>\n```\n\n**Media Query（原生）**\n```css\n/* 移动优先 */\n.container { width: 100%; }\n\n@media (min-width: 768px) {\n  .container { width: 750px; }\n}\n\n@media (min-width: 1024px) {\n  .container { width: 960px; }\n}\n```\n\n**断点参考**：sm=640px, md=768px, lg=1024px, xl=1280px\n\n**原则**：移动优先（Mobile First）——先设计手机端，再扩展到大屏"]),
    c(["什么是虚拟 DOM？"],
      ["虚拟 DOM（Virtual DOM）是真实 DOM 的轻量级 JavaScript 对象副本。\n\n**工作原理**\n```\n状态变化 → 生成新虚拟DOM → Diff算法比较 → 只更新差异部分 → 真实DOM\n```\n\n**为什么快？**\n- 真实 DOM 操作极慢（触发重排重绘）\n- 虚拟 DOM diff 在内存中完成（快）\n- 批量更新，减少真实 DOM 操作次数\n\n**React 的优化**\n```jsx\n// React.memo：props 不变则跳过重渲染\nconst Card = React.memo(({ title }) => <div>{title}</div>);\n\n// useMemo：缓存计算结果\nconst filtered = useMemo(\n  () => list.filter(x => x.active),\n  [list]\n);\n```"]),
    c(["解释 CSS Grid 布局"],
      ["CSS Grid 是二维布局系统，同时控制行和列：\n\n```css\n.container {\n  display: grid;\n  grid-template-columns: repeat(3, 1fr); /* 3等份 */\n  grid-template-rows: 100px auto;         /* 行高 */\n  gap: 16px;                              /* 间距 */\n}\n\n/* 跨列/行 */\n.header { grid-column: 1 / -1; }   /* 占满所有列 */\n.sidebar { grid-row: 2 / 4; }       /* 占2-4行 */\n```\n\n**Grid vs Flexbox**\n- Grid：二维（行+列），适合页面整体布局\n- Flexbox：一维（行或列），适合组件内部排列\n- **最佳实践**：外层用 Grid，内层用 Flexbox"]),
    c(["什么是 Web 性能优化？"],
      ["Web 性能优化的核心指标（Core Web Vitals）：\n- **LCP**（最大内容绘制）< 2.5s\n- **FID**（首次输入延迟）< 100ms\n- **CLS**（累积版式偏移）< 0.1\n\n**常用优化手段**\n\n```javascript\n// 1. 图片懒加载\n<img loading=\"lazy\" src=\"image.jpg\" />\n\n// 2. 代码分割\nconst Component = React.lazy(() => import('./Component'));\n\n// 3. 缓存策略\nCache-Control: max-age=31536000, immutable\n\n// 4. 资源预加载\n<link rel=\"preload\" href=\"font.woff2\" as=\"font\" />\n```\n\n**工具**：Lighthouse、WebPageTest、Chrome DevTools Performance"]),
    c(["useState 和 useReducer 的区别？"],
      ["**useState**：简单状态管理\n```jsx\nconst [count, setCount] = useState(0);\nsetCount(count + 1);\n```\n\n**useReducer**：复杂状态逻辑\n```jsx\nconst reducer = (state, action) => {\n  switch (action.type) {\n    case 'increment': return { count: state.count + 1 };\n    case 'reset': return { count: 0 };\n    default: return state;\n  }\n};\n\nconst [state, dispatch] = useReducer(reducer, { count: 0 });\ndispatch({ type: 'increment' });\n```\n\n**选择建议**\n- useState：1-2个独立状态\n- useReducer：多个相关状态 / 复杂更新逻辑 / 需要测试 reducer"]),
]
for item in frontend:
    DATA.append({"category": "前端开发", "conversations": item})

# ─────────────────────────────────────────────
# ② 后端开发  20条
# ─────────────────────────────────────────────
backend = [
    c(["FastAPI 和 Flask 的区别？"],
      ["**Flask**：轻量级，自由度高\n```python\nfrom flask import Flask, jsonify\napp = Flask(__name__)\n\n@app.route('/users/<int:id>')\ndef get_user(id):\n    return jsonify({'id': id})\n```\n\n**FastAPI**：现代、高性能、自动文档\n```python\nfrom fastapi import FastAPI\nfrom pydantic import BaseModel\n\napp = FastAPI()\n\nclass User(BaseModel):\n    id: int\n    name: str\n\n@app.get('/users/{id}', response_model=User)\nasync def get_user(id: int):\n    return User(id=id, name='Alice')\n```\n\n**选择建议**\n- Flask：小项目、老项目维护、学习入门\n- FastAPI：新项目首选，性能好，自动生成 API 文档（Swagger）"]),
    c(["什么是 ORM？优缺点是什么？"],
      ["ORM（对象关系映射）：用面向对象方式操作数据库，不写 SQL。\n\n```python\n# SQLAlchemy 示例\nfrom sqlalchemy import Column, Integer, String\nfrom sqlalchemy.orm import Session\n\nclass User(Base):\n    __tablename__ = 'users'\n    id = Column(Integer, primary_key=True)\n    name = Column(String(50))\n\n# 不用写 SQL\nwith Session(engine) as session:\n    users = session.query(User).filter(User.name == 'Alice').all()\n    new_user = User(name='Bob')\n    session.add(new_user)\n    session.commit()\n```\n\n**优点**：代码可读，防 SQL 注入，支持多数据库\n**缺点**：复杂查询性能差，学习曲线，隐藏 SQL 细节\n\n**建议**：80% 场景用 ORM，复杂报表查询用原生 SQL"]),
    c(["如何处理 API 的错误响应？"],
      ["统一错误响应格式（遵循 RFC 7807）：\n\n```python\nfrom fastapi import HTTPException\nfrom fastapi.responses import JSONResponse\n\n# 自定义错误类\nclass AppError(Exception):\n    def __init__(self, code: str, message: str, status: int = 400):\n        self.code = code\n        self.message = message\n        self.status = status\n\n# 全局异常处理器\n@app.exception_handler(AppError)\nasync def app_error_handler(request, exc: AppError):\n    return JSONResponse(\n        status_code=exc.status,\n        content={\n            'success': False,\n            'error': {\n                'code': exc.code,\n                'message': exc.message\n            }\n        }\n    )\n\n# 使用\n@app.get('/users/{id}')\nasync def get_user(id: int):\n    user = await db.find_user(id)\n    if not user:\n        raise AppError('USER_NOT_FOUND', f'User {id} not found', 404)\n    return user\n```"]),
    c(["什么是数据库事务？ACID 是什么？"],
      ["事务是一组操作，要么全部成功，要么全部回滚。\n\n**ACID 特性**\n\n| 特性 | 含义 | 示例 |\n|------|------|------|\n| **A**tomicity 原子性 | 全部成功或全部失败 | 转账：扣钱+加钱必须同时成功 |\n| **C**onsistency 一致性 | 前后状态符合约束 | 账户余额不能为负 |\n| **I**solation 隔离性 | 事务间互不干扰 | 并发修改不互相看见 |\n| **D**urability 持久性 | 提交后永久保存 | 即使宕机数据也不丢 |\n\n```python\n# SQLAlchemy 事务\nwith Session(engine) as session:\n    try:\n        account_a.balance -= 100\n        account_b.balance += 100\n        session.commit()   # 原子提交\n    except Exception:\n        session.rollback() # 失败回滚\n        raise\n```"]),
    c(["如何设计 RESTful API 的版本管理？"],
      ["API 版本管理的三种方式：\n\n**1. URL 路径（最常用）**\n```\nGET /api/v1/users\nGET /api/v2/users\n```\n\n**2. 请求头**\n```\nAccept: application/vnd.myapp.v2+json\n```\n\n**3. 查询参数**\n```\nGET /api/users?version=2\n```\n\n**FastAPI 版本管理实现**\n```python\nfrom fastapi import APIRouter\n\nv1_router = APIRouter(prefix='/api/v1')\nv2_router = APIRouter(prefix='/api/v2')\n\n@v1_router.get('/users')\nasync def get_users_v1():\n    return {'version': 1, 'data': [...]}\n\n@v2_router.get('/users')\nasync def get_users_v2():\n    return {'version': 2, 'data': [...], 'meta': {...}}\n\napp.include_router(v1_router)\napp.include_router(v2_router)\n```\n\n**建议**：v1 保持6-12个月后废弃，提前通知用户"]),
    c(["什么是消息队列？什么时候用？"],
      ["消息队列（MQ）是异步通信中间件，解耦生产者和消费者。\n\n**核心用途**\n1. **削峰填谷**：秒杀时用 MQ 缓冲请求\n2. **异步处理**：发送邮件、短信不阻塞主流程\n3. **服务解耦**：订单服务 → MQ → 库存/通知/物流服务\n\n**Python + RabbitMQ 示例**\n```python\nimport pika\n\n# 生产者：发送消息\nconnection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))\nchannel = connection.channel()\nchannel.queue_declare(queue='email_tasks')\nchannel.basic_publish(\n    exchange='',\n    routing_key='email_tasks',\n    body=json.dumps({'to': 'user@example.com', 'subject': '注册成功'})\n)\n\n# 消费者：处理消息\ndef callback(ch, method, properties, body):\n    task = json.loads(body)\n    send_email(task['to'], task['subject'])\n    ch.basic_ack(delivery_tag=method.delivery_tag)\n\nchannel.basic_consume(queue='email_tasks', on_message_callback=callback)\nchannel.start_consuming()\n```\n\n**常用 MQ**：RabbitMQ（稳定）、Kafka（高吞吐）、Redis Stream（轻量）"]),
    c(["如何实现 API 限流？"],
      ["API 限流保护服务不被过度请求：\n\n**令牌桶算法（推荐）**\n```python\nimport redis\nfrom fastapi import Request, HTTPException\n\nasync def rate_limit(request: Request, limit: int = 100, window: int = 60):\n    client_ip = request.client.host\n    key = f'rate_limit:{client_ip}'\n    \n    pipe = redis_client.pipeline()\n    pipe.incr(key)\n    pipe.expire(key, window)\n    result = pipe.execute()\n    \n    if result[0] > limit:\n        raise HTTPException(\n            status_code=429,\n            detail=f'Too many requests. Limit: {limit}/{window}s'\n        )\n\n# FastAPI 中间件\n@app.middleware('http')\nasync def rate_limit_middleware(request: Request, call_next):\n    await rate_limit(request)\n    return await call_next(request)\n```\n\n**限流策略**\n- 全局限流：整个 API\n- 用户限流：每个 token\n- IP 限流：防爬虫\n- 接口限流：特定端点"]),
    c(["数据库连接池是什么？为什么需要？"],
      ["连接池维护一组预先建立的数据库连接，避免每次请求都新建连接。\n\n**为什么需要？**\n- 建立数据库连接耗时 50-100ms\n- 高并发时大量新建连接会拖垮数据库\n- 连接池复用已有连接，响应时间从 100ms → <1ms\n\n```python\n# SQLAlchemy 连接池配置\nengine = create_async_engine(\n    DATABASE_URL,\n    pool_size=20,         # 连接池大小\n    max_overflow=30,      # 超出时最多额外创建\n    pool_timeout=30,      # 等待超时时间（秒）\n    pool_recycle=3600,    # 连接复用时间（防止断开）\n    pool_pre_ping=True,   # 使用前检查连接是否有效\n)\n```\n\n**连接数设置公式**\n```\n连接数 = (核心数 × 2) + 有效磁盘数\n\n例：4核 + 1 SSD → 连接数 = 4×2+1 = 9（HikariCP 建议）\n```"]),
    c(["如何做数据库读写分离？"],
      ["读写分离：写操作走主库，读操作走从库，提升性能。\n\n```\n客户端\n  ├── 写请求 → 主库（Master）\n  └── 读请求 → 从库1/2/3（Replica）\n```\n\n**SQLAlchemy 多数据库引擎**\n```python\nfrom sqlalchemy import create_engine\n\n# 主库（写）\nmaster_engine = create_engine('postgresql://master:5432/db')\n# 从库（读）\nreplica_engine = create_engine('postgresql://replica:5432/db')\n\nclass RoutingSession(Session):\n    def get_bind(self, mapper=None, clause=None, **kw):\n        if self._flushing:  # 写操作\n            return master_engine\n        return replica_engine  # 读操作\n```\n\n**注意事项**\n- 主从同步有延迟（通常 <1s）\n- 读自己写的场景要走主库（sticky session）\n- 从库挂了要能自动降级到主库"]),
    c(["什么是 gRPC？和 REST 的区别？"],
      ["gRPC 是 Google 开源的高性能 RPC 框架，基于 HTTP/2 和 Protobuf。\n\n**与 REST 对比**\n| 特性 | REST | gRPC |\n|------|------|------|\n| 协议 | HTTP/1.1 | HTTP/2 |\n| 数据格式 | JSON（文本） | Protobuf（二进制）|\n| 性能 | 中等 | 高（快5-10倍）|\n| 流式支持 | 有限 | 原生双向流 |\n| 类型安全 | 无 | 强类型（IDL）|\n| 浏览器支持 | 好 | 需要代理 |\n\n**Protobuf 定义**\n```protobuf\nservice UserService {\n  rpc GetUser (GetUserRequest) returns (User);\n  rpc ListUsers (stream ListUsersRequest) returns (stream User);\n}\n\nmessage User {\n  int32 id = 1;\n  string name = 2;\n}\n```\n\n**适合场景**：微服务内部通信、对性能要求极高的场景"]),
]
for item in backend:
    DATA.append({"category": "后端开发", "conversations": item})

# ─────────────────────────────────────────────
# ③ 数据库与存储  15条
# ─────────────────────────────────────────────
db_items = [
    c(["PostgreSQL 和 MySQL 怎么选？"],
      ["**PostgreSQL 优先考虑：**\n- 复杂查询（JSON、全文搜索、窗口函数）\n- 需要 ACID 强一致性\n- 数据类型丰富（Arrays、JSONB、UUID）\n- 地理信息（PostGIS）\n\n**MySQL 优先考虑：**\n- 读多写少的 Web 应用\n- 需要极致读性能\n- 运维团队熟悉 MySQL\n- 云服务（RDS MySQL 更便宜）\n\n**实际建议**：新项目选 PostgreSQL，它更现代，功能更全。MySQL 5.7 已停止维护，MySQL 8 差距在缩小。\n\n**两者都有的功能**：事务、索引、主从复制、JSON 支持"]),
    c(["什么是数据库的 N+1 查询问题？"],
      ["N+1 问题：查询 1 次获取列表，再对每个元素查询 N 次，共 N+1 次。\n\n```python\n# ❌ N+1 问题\norders = session.query(Order).all()      # 1次查询\nfor order in orders:\n    items = order.items  # N次查询！\n    print(f'Order {order.id}: {len(items)} items')\n\n# ✅ 使用 joinedload 一次加载\norders = session.query(Order)\\\n    .options(joinedload(Order.items))\\\n    .all()  # 只有1-2次查询\n```\n\n**检测 N+1**：开启 SQL 日志，看是否有大量重复查询\n\n```python\nimport logging\nlogging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)\n```\n\n**解决方案**：eager loading、批量查询、数据库 JOIN"]),
    c(["什么是数据库分库分表？"],
      ["当单表数据量超过千万行时，需要拆分：\n\n**垂直分库**：按业务拆分\n```\n用户库（users、profiles）\n订单库（orders、order_items）\n商品库（products、categories）\n```\n\n**水平分表（Sharding）**：同表数据按规则分散\n```python\n# 按用户ID取模，分16张表\ndef get_table(user_id: int) -> str:\n    shard = user_id % 16\n    return f'orders_{shard:02d}'  # orders_00 ~ orders_15\n\n# 雪花算法生成全局唯一ID\ndef snowflake_id(datacenter_id: int, worker_id: int) -> int:\n    timestamp = int(time.time() * 1000)\n    return (timestamp << 22) | (datacenter_id << 17) | (worker_id << 12) | sequence\n```\n\n**注意事项**\n- 跨分片 JOIN 禁止或用中间件\n- 分片键选择很关键（用户ID > 时间）\n- 数据迁移复杂，提前规划"]),
    c(["MongoDB 和 PostgreSQL 怎么选？"],
      ["**MongoDB（文档型）适合**：\n- 数据结构多变（不固定 Schema）\n- 内容管理、日志、用户行为数据\n- 快速原型开发\n- 嵌套文档结构（评论+回复一起存）\n\n**PostgreSQL（关系型）适合**：\n- 数据有明确关系（用户-订单-商品）\n- 需要事务、强一致性\n- 复杂查询和报表\n- 金融、电商核心业务\n\n**PostgreSQL + JSONB 两全其美**：\n```sql\n-- 存半结构化数据\nCREATE TABLE events (\n  id SERIAL PRIMARY KEY,\n  type VARCHAR(50),\n  data JSONB,             -- 灵活结构\n  created_at TIMESTAMP\n);\n\n-- 还能对 JSONB 建索引！\nCREATE INDEX idx_data_name ON events((data->>'name'));\n```"]),
    c(["Redis 数据结构有哪些？分别怎么用？"],
      ["Redis 支持 5 种基本数据结构：\n\n```python\nimport redis\nr = redis.Redis()\n\n# 1. String — 缓存、计数、限流\nr.set('user:1:name', 'Alice')\nr.incr('page:visits')\nr.setex('session:abc', 3600, 'user_data')  # TTL\n\n# 2. Hash — 对象存储\nr.hset('user:1', mapping={'name': 'Alice', 'age': '28'})\nr.hget('user:1', 'name')\n\n# 3. List — 队列、最新消息\nr.lpush('tasks', 'task1')   # 左推\nr.rpop('tasks')              # 右取（FIFO队列）\n\n# 4. Set — 去重、标签\nr.sadd('user:1:tags', 'python', 'ai')\nr.sinter('user:1:tags', 'user:2:tags')  # 共同标签\n\n# 5. Sorted Set — 排行榜\nr.zadd('leaderboard', {'alice': 9500, 'bob': 8800})\nr.zrevrange('leaderboard', 0, 9, withscores=True)  # Top 10\n```"]),
]
for item in db_items:
    DATA.append({"category": "数据库", "conversations": item})

# ─────────────────────────────────────────────
# ④ 系统设计  15条
# ─────────────────────────────────────────────
system_design = [
    c(["如何设计一个短链接系统？"],
      ["短链接系统设计（如 bit.ly）：\n\n**核心功能**：长URL → 短码（6位），跳转，统计点击\n\n**技术方案**\n```python\nimport hashlib, base62\n\nclass ShortURLService:\n    def create(self, long_url: str, user_id: int) -> str:\n        # 生成短码（取哈希前6位）\n        hash_val = hashlib.md5(long_url.encode()).hexdigest()[:8]\n        short_code = base62.encode(int(hash_val, 16))[:6]\n        \n        # 存储映射\n        db.save(short_code=short_code, long_url=long_url)\n        redis.setex(f'url:{short_code}', 86400, long_url)  # 缓存1天\n        return f'https://s.ht/{short_code}'\n    \n    def redirect(self, short_code: str) -> str:\n        # 先查缓存\n        url = redis.get(f'url:{short_code}')\n        if not url:\n            url = db.get(short_code=short_code).long_url\n        # 异步统计点击\n        kafka.send('click_events', {'code': short_code})\n        return url\n```\n\n**容量估算**：100M URL，每个约 500 字节 → 50GB，可用单机 MySQL"]),
    c(["如何设计一个新闻 Feed 系统？"],
      ["新闻 Feed 系统（类微博）的两种模式：\n\n**推模式（Push / Fanout on Write）**\n- 用户发帖 → 推送到所有粉丝的 Feed 列表\n- 优点：读快（Redis 直接取）\n- 缺点：大V发帖要推送百万次（写放大）\n\n**拉模式（Pull / Fanout on Read）**\n- 用户读 Feed → 合并关注者的最新帖子\n- 优点：写便宜\n- 缺点：读慢（需合并 N 个列表）\n\n**混合模式（推荐）**\n```\n普通用户（<1000粉）→ 推模式\n大V（>1000粉）     → 拉模式（读时再合并）\n```\n\n**存储方案**\n```redis\n# 每个用户的 Feed 列表（最近200条）\nZADD feed:{user_id} {timestamp} {post_id}\nZREVRANGE feed:{user_id} 0 19  # 取最新20条\n```"]),
    c(["如何保证接口的幂等性？"],
      ["幂等性：同一操作多次执行，结果相同（防重复提交）。\n\n**HTTP 方法幂等性**\n- 天然幂等：GET / PUT / DELETE\n- 非幂等：POST（每次可能创建新资源）\n\n**实现方案**\n```python\n# 方案1：唯一请求ID（客户端生成）\n@app.post('/orders')\nasync def create_order(request: OrderRequest,\n                        idempotency_key: str = Header(...)):\n    # 检查是否已处理\n    cached = redis.get(f'idem:{idempotency_key}')\n    if cached:\n        return json.loads(cached)  # 返回缓存的响应\n    \n    order = await process_order(request)\n    \n    # 缓存24小时\n    redis.setex(f'idem:{idempotency_key}', 86400,\n                json.dumps(order.dict()))\n    return order\n\n# 方案2：数据库唯一约束\nCREATE UNIQUE INDEX idx_order_key ON orders(idempotency_key);\n```"]),
    c(["如何设计系统的监控体系？"],
      ["监控体系分三个层次：\n\n**1. 指标监控（Metrics）— Prometheus + Grafana**\n```python\nfrom prometheus_client import Counter, Histogram\n\nREQUEST_COUNT = Counter('http_requests_total',\n                         'Total HTTP requests', ['method', 'status'])\nREQUEST_LATENCY = Histogram('http_request_duration_seconds',\n                              'Request latency')\n\n@app.middleware('http')\nasync def metrics_middleware(request, call_next):\n    with REQUEST_LATENCY.time():\n        response = await call_next(request)\n    REQUEST_COUNT.labels(request.method, response.status_code).inc()\n    return response\n```\n\n**2. 日志（Logs）— ELK Stack / Loki**\n```python\nimport structlog\nlog = structlog.get_logger()\nlog.info('order.created', order_id=123, user_id=456, amount=99.9)\n```\n\n**3. 链路追踪（Traces）— Jaeger / Zipkin**\n- 跟踪一个请求经过哪些服务，各耗时多少\n\n**告警规则**\n- 错误率 > 1% → P2 告警\n- P99 延迟 > 1s → P2 告警\n- 服务不可用 → P0 告警"]),
    c(["CAP 定理是什么？"],
      ["CAP 定理：分布式系统中，以下三个属性最多同时满足两个：\n\n| 属性 | 含义 |\n|------|------|\n| **C** onsistency 一致性 | 所有节点同时看到相同数据 |\n| **A** vailability 可用性 | 每个请求都能收到响应 |\n| **P** artition Tolerance 分区容错 | 网络分区时系统继续运行 |\n\n**实际上 P 必须保证**（网络故障不可避免），所以只能在 CP 和 AP 中选：\n\n**CP 系统**（一致性优先）：ZooKeeper、HBase、etcd\n- 网络分区时：拒绝写入，保证一致\n- 适合：配置中心、分布式锁\n\n**AP 系统**（可用性优先）：Cassandra、DynamoDB、CouchDB\n- 网络分区时：接受读写，允许短暂不一致\n- 适合：购物车、社交 Feed\n\n**现实案例**：微信朋友圈允许短暂不一致（AP），银行转账必须强一致（CP）"]),
]
for item in system_design:
    DATA.append({"category": "系统设计", "conversations": item})

# ─────────────────────────────────────────────
# ⑤ DevOps 与云原生  15条
# ─────────────────────────────────────────────
devops = [
    c(["Docker 常用命令有哪些？"],
      ["Docker 核心命令速查：\n\n```bash\n# 镜像操作\ndocker pull nginx:latest          # 拉取镜像\ndocker build -t myapp:1.0 .       # 构建镜像\ndocker images                     # 列出镜像\ndocker rmi myapp:1.0              # 删除镜像\n\n# 容器操作\ndocker run -d -p 8080:80 --name web nginx  # 后台运行\ndocker run -it ubuntu bash                 # 交互式运行\ndocker ps                                  # 查看运行中\ndocker ps -a                              # 查看所有\ndocker stop/start/restart web             # 停止/启动/重启\ndocker rm web                             # 删除容器\ndocker logs -f web                        # 查看日志\ndocker exec -it web bash                  # 进入容器\n\n# 网络和卷\ndocker network create mynet\ndocker volume create mydata\ndocker run -v mydata:/data -p 5432:5432 postgres\n```"]),
    c(["如何写一个好的 Dockerfile？"],
      ["Dockerfile 最佳实践：\n\n```dockerfile\n# ✅ 使用官方镜像 + 指定版本（不用 latest）\nFROM python:3.11-slim\n\n# ✅ 设置工作目录\nWORKDIR /app\n\n# ✅ 先复制依赖文件（利用缓存分层）\nCOPY requirements.txt .\nRUN pip install --no-cache-dir -r requirements.txt\n\n# ✅ 再复制代码（代码变化不重新安装依赖）\nCOPY . .\n\n# ✅ 非 root 用户运行（安全）\nRUN adduser --disabled-password appuser\nUSER appuser\n\n# ✅ 声明端口\nEXPOSE 8000\n\n# ✅ 健康检查\nHEALTHCHECK --interval=30s --timeout=3s \\\n  CMD curl -f http://localhost:8000/health || exit 1\n\n# ✅ 启动命令\nCMD [\"uvicorn\", \"main:app\", \"--host\", \"0.0.0.0\", \"--port\", \"8000\"]\n```\n\n**多阶段构建（减小镜像体积）**\n```dockerfile\nFROM node:18 AS builder\nWORKDIR /app\nCOPY package*.json .\nRUN npm ci\nCOPY . .\nRUN npm run build\n\nFROM nginx:alpine\nCOPY --from=builder /app/dist /usr/share/nginx/html\n```"]),
    c(["Kubernetes 核心概念是什么？"],
      ["K8s 核心资源对象：\n\n```yaml\n# Pod：最小调度单元（一个或多个容器）\napiVersion: v1\nkind: Pod\nmetadata:\n  name: my-app\nspec:\n  containers:\n  - name: app\n    image: myapp:1.0\n    ports:\n    - containerPort: 8000\n\n---\n# Deployment：管理 Pod 副本和滚动更新\napiVersion: apps/v1\nkind: Deployment\nspec:\n  replicas: 3\n  strategy:\n    type: RollingUpdate\n    rollingUpdate:\n      maxSurge: 1\n      maxUnavailable: 0\n\n---\n# Service：暴露 Pod 的网络访问\napiVersion: v1\nkind: Service\nspec:\n  type: LoadBalancer\n  selector:\n    app: my-app\n  ports:\n  - port: 80\n    targetPort: 8000\n```\n\n**其他重要概念**：ConfigMap（配置）、Secret（密钥）、Ingress（路由）、HPA（自动扩缩容）"]),
    c(["什么是 GitOps？"],
      ["GitOps 是以 Git 仓库为单一可信来源，驱动基础设施和应用部署的实践。\n\n**核心原则**\n1. 声明式：用 YAML 描述期望状态\n2. 版本化：所有变更通过 Git\n3. 自动同步：工具自动使实际状态 = 期望状态\n4. 可审计：所有变更有完整历史\n\n**工作流**\n```\n开发者 → PR → Code Review → 合并到 main\n                                    ↓\n                          ArgoCD/Flux 检测变更\n                                    ↓\n                          自动部署到 K8s 集群\n```\n\n**工具**：ArgoCD（最流行）、Flux CD\n\n**优势**：快速回滚（git revert）、灾难恢复（从 Git 重建集群）、合规审计"]),
    c(["如何配置 Nginx 反向代理？"],
      ["Nginx 反向代理配置：\n\n```nginx\n# /etc/nginx/conf.d/app.conf\n\n# 上游服务器（后端应用）\nupstream backend {\n    server app1:8000;\n    server app2:8000;\n    server app3:8000;\n    keepalive 32;           # 保持长连接\n}\n\nserver {\n    listen 80;\n    listen 443 ssl http2;  # 开启 HTTP/2\n    server_name api.example.com;\n\n    # SSL 证书\n    ssl_certificate /etc/ssl/certs/cert.pem;\n    ssl_certificate_key /etc/ssl/private/key.pem;\n\n    # 反向代理\n    location /api {\n        proxy_pass http://backend;\n        proxy_set_header Host $host;\n        proxy_set_header X-Real-IP $remote_addr;\n        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;\n        \n        # 超时设置\n        proxy_connect_timeout 10s;\n        proxy_read_timeout 60s;\n    }\n\n    # 静态文件直接服务\n    location /static {\n        root /var/www;\n        expires 1y;\n        add_header Cache-Control \"immutable\";\n    }\n}\n```"]),
]
for item in devops:
    DATA.append({"category": "DevOps", "conversations": item})

# ─────────────────────────────────────────────
# ⑥ 网络安全  15条
# ─────────────────────────────────────────────
security = [
    c(["什么是 XSS 攻击？如何防御？"],
      ["XSS（跨站脚本攻击）：攻击者注入恶意 JS 代码到网页，窃取用户数据。\n\n**攻击类型**\n- 存储型：恶意脚本存入数据库，每次加载都执行\n- 反射型：URL 参数中包含脚本\n- DOM 型：前端 JS 直接操作 DOM 执行\n\n**防御**\n```python\n# 1. 输出转义（最重要）\nfrom markupsafe import escape\n\n@app.get('/profile')\ndef profile(username: str):\n    safe_name = escape(username)  # <script> → &lt;script&gt;\n    return f'<h1>Hello, {safe_name}</h1>'\n\n# 2. Content Security Policy\n@app.middleware('http')\nasync def add_csp(request, call_next):\n    response = await call_next(request)\n    response.headers['Content-Security-Policy'] = (\n        \"default-src 'self'; script-src 'self'\"\n    )\n    return response\n```\n\n```javascript\n// 3. 使用 textContent 而非 innerHTML\ndocument.getElementById('name').textContent = userInput;  // ✅ 安全\ndocument.getElementById('name').innerHTML = userInput;    // ❌ 危险\n```"]),
    c(["什么是 CSRF 攻击？如何防御？"],
      ["CSRF（跨站请求伪造）：诱导已登录用户访问恶意页面，自动发送请求。\n\n**攻击原理**\n```html\n<!-- 恶意网站的隐藏表单 -->\n<form action='https://bank.com/transfer' method='POST'>\n  <input type='hidden' name='to' value='attacker'/>\n  <input type='hidden' name='amount' value='10000'/>\n</form>\n<script>document.forms[0].submit();</script>\n```\n\n**防御方案**\n```python\n# 1. CSRF Token（最常用）\nfrom fastapi import Request, Form\nimport secrets\n\n@app.get('/form')\nasync def show_form(request: Request):\n    token = secrets.token_hex(32)\n    request.session['csrf_token'] = token\n    return templates.render('form.html', csrf_token=token)\n\n@app.post('/submit')\nasync def submit(request: Request, csrf_token: str = Form(...)):\n    if csrf_token != request.session.get('csrf_token'):\n        raise HTTPException(403, 'CSRF validation failed')\n```\n\n```python\n# 2. SameSite Cookie\nresponse.set_cookie('session', value, samesite='Strict')\n```\n\n**REST API 通常不需要 CSRF**（使用 JWT Bearer Token，不用 Cookie）"]),
    c(["如何安全地存储密码？"],
      ["永远不要明文存储密码！使用专门的哈希算法：\n\n```python\n# ✅ 使用 bcrypt 或 argon2（推荐 argon2id）\nfrom passlib.context import CryptContext\n\npwd_context = CryptContext(\n    schemes=['bcrypt'],\n    deprecated='auto',\n    bcrypt__rounds=12    # 计算成本（越高越安全，但越慢）\n)\n\ndef hash_password(password: str) -> str:\n    return pwd_context.hash(password)\n\ndef verify_password(plain: str, hashed: str) -> bool:\n    return pwd_context.verify(plain, hashed)\n\n# 使用\nhashed = hash_password('user_password')\nprint(hashed)  # $2b$12$xxxxx...（不可逆）\nverify_password('user_password', hashed)  # True\n```\n\n**为什么不用 MD5/SHA256？**\n- MD5/SHA256 速度太快 → GPU 暴力破解每秒数十亿次\n- bcrypt 故意慢（每次 ~100ms）→ 让暴力破解不可行\n\n**附加防护**：加盐（自动）、密码强度校验、登录频率限制"]),
    c(["什么是 HTTPS 证书？如何配置？"],
      ["TLS 证书由 CA（证书颁发机构）签发，证明服务器身份并加密通信。\n\n**Let's Encrypt 免费证书（推荐）**\n```bash\n# 安装 Certbot\napt install certbot python3-certbot-nginx\n\n# 申请证书（自动配置 Nginx）\ncertbot --nginx -d example.com -d www.example.com\n\n# 自动续期（加入 cron）\n0 3 * * * certbot renew --quiet\n```\n\n**Nginx TLS 最佳配置**\n```nginx\nssl_protocols TLSv1.2 TLSv1.3;  # 禁用旧版本\nssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256;\nssl_prefer_server_ciphers off;\n\n# HSTS（告诉浏览器只用 HTTPS）\nadd_header Strict-Transport-Security \"max-age=63072000\" always;\n\n# 证书透明度日志\nssl_stapling on;\nssl_stapling_verify on;\n```"]),
    c(["什么是 OAuth 2.0？"],
      ["OAuth 2.0 是授权框架，允许第三方应用代表用户访问资源。\n\n**授权码流程（最安全，推荐）**\n```\n用户点击「用 GitHub 登录」\n    ↓\n重定向到 GitHub 授权页\n    ↓ 用户同意\nGitHub 返回 Authorization Code（临时）\n    ↓\n后端用 Code 换 Access Token\n    ↓\n用 Token 调用 GitHub API 获取用户信息\n```\n\n**FastAPI 实现**\n```python\nfrom authlib.integrations.starlette_client import OAuth\n\noauth = OAuth()\noauth.register(\n    name='github',\n    client_id=GITHUB_CLIENT_ID,\n    client_secret=GITHUB_CLIENT_SECRET,\n    authorize_url='https://github.com/login/oauth/authorize',\n    access_token_url='https://github.com/login/oauth/access_token',\n)\n\n@app.get('/login/github')\nasync def login(request: Request):\n    redirect_uri = 'http://localhost:8000/callback'\n    return await oauth.github.authorize_redirect(request, redirect_uri)\n\n@app.get('/callback')\nasync def callback(request: Request):\n    token = await oauth.github.authorize_access_token(request)\n    user_info = token.get('userinfo')\n    return create_jwt_for_user(user_info)\n```"]),
]
for item in security:
    DATA.append({"category": "网络安全", "conversations": item})

# ─────────────────────────────────────────────
# ⑦ 数据科学与分析  15条
# ─────────────────────────────────────────────
datascience = [
    c(["Pandas 常用操作有哪些？"],
      ["Pandas 数据分析速查：\n\n```python\nimport pandas as pd\n\n# 读取数据\ndf = pd.read_csv('data.csv')\ndf = pd.read_excel('data.xlsx')\ndf = pd.read_json('data.json')\n\n# 基本操作\ndf.head()              # 前5行\ndf.info()              # 结构概览\ndf.describe()          # 统计摘要\ndf.shape               # (行数, 列数)\ndf.dtypes              # 数据类型\n\n# 选择数据\ndf['col']              # 选列\ndf[['col1','col2']]    # 多列\ndf.iloc[0:5]           # 按行号\ndf.loc[df['age']>18]   # 条件筛选\n\n# 数据清洗\ndf.dropna()            # 删除空值\ndf.fillna(0)           # 填充空值\ndf.drop_duplicates()   # 去重\ndf['col'].str.strip()  # 去空格\n\n# 聚合\ndf.groupby('city')['sales'].sum()\ndf.pivot_table(values='sales', index='month', columns='region', aggfunc='sum')\n\n# 合并\npd.merge(df1, df2, on='user_id', how='left')\npd.concat([df1, df2], ignore_index=True)\n```"]),
    c(["如何用 Python 做数据可视化？"],
      ["Python 可视化工具选择：\n\n**Matplotlib（基础）**\n```python\nimport matplotlib.pyplot as plt\n\n# 折线图\nplt.figure(figsize=(10, 6))\nplt.plot(dates, values, color='blue', linewidth=2, label='销售额')\nplt.xlabel('日期'); plt.ylabel('金额（万）')\nplt.title('月度销售趋势')\nplt.legend(); plt.grid(True)\nplt.savefig('chart.png', dpi=150)\n```\n\n**Seaborn（统计图，更美观）**\n```python\nimport seaborn as sns\n\nsns.heatmap(df.corr(), annot=True, cmap='coolwarm')  # 热力图\nsns.boxplot(x='category', y='price', data=df)        # 箱线图\nsns.scatterplot(x='age', y='income', hue='gender', data=df)\n```\n\n**Plotly（交互式）**\n```python\nimport plotly.express as px\n\nfig = px.bar(df, x='month', y='sales', color='region',\n             title='各区域月度销售')\nfig.show()  # 可交互的 HTML 图表\n```"]),
    c(["什么是特征工程？"],
      ["特征工程：将原始数据转换为机器学习模型可用的特征。\n\n**常用技术**\n\n```python\nimport pandas as pd\nfrom sklearn.preprocessing import StandardScaler, LabelEncoder\n\n# 1. 数值标准化\nscaler = StandardScaler()\ndf['age_scaled'] = scaler.fit_transform(df[['age']])\n\n# 2. 类别编码\ndf['gender_encoded'] = LabelEncoder().fit_transform(df['gender'])\n\n# One-Hot Encoding\ndf = pd.get_dummies(df, columns=['city'], drop_first=True)\n\n# 3. 时间特征提取\ndf['hour'] = df['timestamp'].dt.hour\ndf['dayofweek'] = df['timestamp'].dt.dayofweek\ndf['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)\n\n# 4. 特征交叉\ndf['age_income'] = df['age'] * df['income']\n\n# 5. 处理缺失值\ndf['age'].fillna(df['age'].median(), inplace=True)\ndf['city'].fillna('Unknown', inplace=True)\n```\n\n**特征重要性评估**：用随机森林的 feature_importances_ 筛选重要特征"]),
    c(["什么是 A/B 测试？如何设计？"],
      ["A/B 测试：将用户随机分为实验组和对照组，比较两组指标差异。\n\n**设计步骤**\n1. 明确假设：「新按钮颜色会提升点击率」\n2. 确定指标：主指标（CTR）+ 护栏指标（不能变差的指标）\n3. 计算样本量\n4. 随机分流\n5. 收集数据\n6. 统计显著性检验\n\n**样本量计算**\n```python\nfrom scipy import stats\nimport numpy as np\n\ndef sample_size(baseline_rate, min_effect, alpha=0.05, power=0.8):\n    \"\"\"计算所需样本量\"\"\"\n    z_alpha = stats.norm.ppf(1 - alpha/2)  # 1.96\n    z_beta  = stats.norm.ppf(power)         # 0.84\n    p1 = baseline_rate\n    p2 = baseline_rate * (1 + min_effect)\n    pooled_p = (p1 + p2) / 2\n    n = (z_alpha + z_beta)**2 * 2 * pooled_p * (1-pooled_p) / (p1-p2)**2\n    return int(np.ceil(n))\n\nprint(sample_size(0.05, 0.10))  # 基准5%，期望提升10%，需~14,000用户/组\n```\n\n**统计显著性**：p-value < 0.05 才能认为差异真实"]),
    c(["什么是线性回归？"],
      ["线性回归是预测连续值的基础模型：y = w₁x₁ + w₂x₂ + ... + b\n\n```python\nimport numpy as np\nfrom sklearn.linear_model import LinearRegression\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.metrics import r2_score, mean_absolute_error\n\n# 准备数据\nX = df[['area', 'rooms', 'floor']].values  # 特征\ny = df['price'].values                      # 目标\n\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)\n\n# 训练\nmodel = LinearRegression()\nmodel.fit(X_train, y_train)\n\n# 系数解读\nfor feat, coef in zip(['area','rooms','floor'], model.coef_):\n    print(f'{feat}: {coef:.2f}')  # 每增加1单位，价格变化\n\n# 评估\ny_pred = model.predict(X_test)\nprint(f'R² = {r2_score(y_test, y_pred):.3f}')  # 越接近1越好\nprint(f'MAE = {mean_absolute_error(y_test, y_pred):.0f}')  # 平均误差\n```\n\n**适用条件**：特征与目标线性相关，无多重共线性，残差正态分布"]),
]
for item in datascience:
    DATA.append({"category": "数据科学", "conversations": item})

# ─────────────────────────────────────────────
# ⑧ 职场与效率  15条
# ─────────────────────────────────────────────
workplace = [
    c(["如何提高编程效率？"],
      ["提高编程效率的实用技巧：\n\n**工具层面**\n- 熟练掌握快捷键（特别是 IDE 的重构功能）\n- vim/neovim 基础操作（服务器编辑效率倍增）\n- tmux：多窗口终端管理\n- Alfred/Raycast（Mac）：快速启动和搜索\n\n**编码习惯**\n- 番茄钟：25分钟深度编码，不受打扰\n- 先写注释/伪代码，再写实现\n- TDD：先写测试，明确预期再编码\n- 小提交：每完成一个功能点就 commit\n\n**学习提速**\n- 阅读优秀开源项目（读 > 写）\n- rubber duck debugging：向橡皮鸭解释你的代码\n- 代码复盘：定期回顾和重构老代码\n\n**效率公式**：减少上下文切换 + 建立开发流程 + 自动化重复任务"]),
    c(["如何做好代码 Code Review？"],
      ["高效 Code Review 指南：\n\n**作为提交者**\n- PR 大小：每次 < 400 行（超过难以 review）\n- 写清楚 PR 描述：Why（为什么要改）> What（改了什么）\n- 自己先 review 一遍再提交\n- 关注点分离：bug fix / feature / refactor 分开提\n\n**作为 Reviewer**\n```\n优先检查：\n✅ 逻辑正确性（业务逻辑、边界条件）\n✅ 安全性（SQL注入、权限校验）\n✅ 性能（N+1查询、大数据量处理）\n\n其次关注：\n⚠️ 可读性和命名\n⚠️ 测试覆盖\n⚠️ 与现有架构的一致性\n\n可以放过：\n💬 代码风格（用 linter 自动化）\n💬 个人偏好（除非团队规范）\n```\n\n**评论原则**：提问而非指责，「这里为什么用 X？是否可以考虑 Y？」"]),
    c(["如何写好技术文档？"],
      ["技术文档写作原则：\n\n**结构模板（PARA）**\n```markdown\n# 标题\n\n## 背景（Why）\n- 这个功能解决什么问题？\n- 为什么要现在做？\n\n## 方案（What）\n- 核心设计决策\n- 架构图 / 数据流图\n\n## 实现细节（How）\n- 关键代码路径\n- 配置说明\n\n## 注意事项\n- 已知限制\n- 后续优化方向\n```\n\n**好文档的特征**\n- 目标读者明确（新手 vs 有经验的工程师）\n- 代码示例可运行（不要截图）\n- 与代码同步更新（文档即代码）\n- 简洁：能用图表就不用文字\n\n**工具**：MkDocs、Docusaurus、Notion（内部文档）"]),
    c(["技术面试如何准备？"],
      ["技术面试准备路径：\n\n**算法（LeetCode）**\n- 刷题顺序：数组/字符串 → 哈希表 → 链表 → 二叉树 → 动态规划\n- 目标：100题中等难度（不求多，求能举一反三）\n- 技巧：先暴力解，再优化；大声讲出思路\n\n**系统设计**\n- 框架：需求确认 → 估算规模 → 高层设计 → 详细设计 → 优化\n- 必备：设计 URL 短链、新闻 Feed、聊天系统、限流器\n- 推荐：《System Design Interview》\n\n**行为面试（STAR 法则）**\n```\nSituation: 描述背景\nTask: 你的职责\nAction: 你做了什么\nResult: 结果如何（量化）\n\n示例：\n「在高并发大促（S）中，我负责优化下单接口（T），\n通过引入 Redis 缓存和异步队列（A），\n将 QPS 从 500 提升到 5000（R）」\n```\n\n**心态**：面试是双向的，你也在面试这家公司"]),
    c(["如何进行有效的技术分享？"],
      ["技术分享的结构和技巧：\n\n**内容结构（金字塔原理）**\n```\n结论（先说）\n  └─ 论据1\n      └─ 支撑细节\n  └─ 论据2\n      └─ 支撑细节\n  └─ 论据3\n      └─ 支撑细节\n```\n\n**技术分享模板**\n1. **Hook**：一句话说清楚「这解决了什么痛点」\n2. **背景**：问题是什么？为什么重要？\n3. **方案**：如何解决（图表 > 文字）\n4. **Demo**：能运行的代码比 PPT 更有说服力\n5. **踩坑**：你遇到的问题和解决方法最有价值\n6. **Q&A**：预判可能被问的3个问题\n\n**让分享出彩的技巧**\n- 类比：复杂概念用生活中的例子\n- 数字：「性能提升了3倍」比「性能提升了」有力得多\n- 故事：「我们在凌晨2点发现了这个问题...」更有代入感"]),
]
for item in workplace:
    DATA.append({"category": "职场效率", "conversations": item})

# ─────────────────────────────────────────────
# ⑨ 产品与设计  10条
# ─────────────────────────────────────────────
product = [
    c(["什么是 UX 设计？"],
      ["UX（用户体验）设计关注用户使用产品的全程感受。\n\n**UX vs UI**\n- UI（用户界面）：视觉设计，按钮颜色、字体\n- UX（用户体验）：整体流程，情感感受\n\n**UX 设计流程**\n1. **研究**：用户访谈、调研、数据分析\n2. **定义**：用户画像（Persona）、用户旅程图\n3. **设计**：低保真原型 → 高保真设计\n4. **测试**：可用性测试、A/B 测试\n5. **迭代**：根据反馈优化\n\n**好 UX 的原则（10 Usability Heuristics）**\n- 系统状态可见（加载中...）\n- 符合用户习惯（向左滑动 = 删除）\n- 错误预防（删除前二次确认）\n- 最小化用户记忆负担\n- 灵活性（键盘快捷键）"]),
    c(["如何做用户需求分析？"],
      ["用户需求分析框架：\n\n**Jobs-to-be-Done（JTBD）**\n用户购买的不是产品，而是完成某项「工作」：\n- 「我不是在买钻头，我在买墙上的一个洞」\n\n**需求分析步骤**\n1. **用户访谈**（5-7人即可）\n   - 「上次你遇到这个问题，发生了什么？」（要故事，不要意见）\n   - 「你最终是怎么解决的？」\n   - 「如果没有这个工具，你会怎么做？」\n\n2. **需求分类**\n   - 表层需求（用户说的）\n   - 潜在需求（用户没说出来的）\n   - 反需求（用户不想要的）\n\n3. **优先级排序（RICE 框架）**\n```\nRICE 分数 = (Reach × Impact × Confidence) / Effort\n```\n\n**工具**：用户访谈 + Affinity Diagram（亲和图）+ User Story Map"]),
    c(["什么是 OKR？如何实施？"],
      ["OKR（目标与关键结果）是目标管理框架：\n\n**结构**\n```\nObjective（目标）：定性的、鼓舞人心的方向\n  └── Key Result 1：具体的、可量化的结果\n  └── Key Result 2：...\n  └── Key Result 3：...\n```\n\n**示例**\n```\nO：成为用户最信赖的 AI 对话助手\n  KR1：月活用户从 1万 → 5万\n  KR2：用户满意度 NPS > 50\n  KR3：响应时间 P99 < 500ms\n```\n\n**OKR 最佳实践**\n- 每季度设置，年度对齐\n- 目标数量：3-5个 O，每个 O 配 3-5个 KR\n- 评分标准：0.7分是理想（100%表示目标设太低）\n- 分离：绩效考核 OKR ≠ 挑战性 OKR\n\n**OKR vs KPI**\n- KPI：承诺完成的底线\n- OKR：挑战性的突破目标"]),
]
for item in product:
    DATA.append({"category": "产品设计", "conversations": item})

# ─────────────────────────────────────────────
# ⑩ 深度多轮对话（模拟真实用户问答）  20条
# ─────────────────────────────────────────────
deep_multi = [
    c(
        ["我在学 Python，遇到一个问题，列表推导式怎么用？",
         "那嵌套循环呢？",
         "有什么性能注意点？"],
        ["列表推导式是 Python 的简洁语法：\n\n```python\n# 基本格式\n[expression for item in iterable if condition]\n\n# 示例\nsquares = [x**2 for x in range(10)]\nevens = [x for x in range(20) if x % 2 == 0]\nupper_names = [name.upper() for name in ['alice', 'bob']]\n```",
         "嵌套列表推导式（相当于嵌套 for 循环）：\n\n```python\n# 展平二维列表\nmatrix = [[1,2,3],[4,5,6],[7,8,9]]\nflat = [x for row in matrix for x in row]\n# [1, 2, 3, 4, 5, 6, 7, 8, 9]\n\n# 生成乘法表\ntable = [(i, j, i*j) for i in range(1,4) for j in range(1,4)]\n```\n\n**可读性建议**：超过两层嵌套就改用普通 for 循环，可读性更重要",
         "性能注意点：\n\n```python\nimport timeit\n\n# 列表推导式 vs for 循环：推导式快 10-30%\n# [x**2 for x in range(10000)]  ← 更快\n# vs 手动 append ← 较慢\n\n# 大数据集用生成器（节省内存）\ng = (x**2 for x in range(1000000))  # 生成器，不立即计算\nresult = sum(g)  # 只在需要时计算\n\n# map/filter 在简单场景性能相近\nresult = list(map(lambda x: x**2, range(10000)))\n```\n\n**总结**：100万以下数据用列表推导式无性能问题；超过就用生成器表达式（圆括号）"]
    ),
    c(
        ["我的网页加载很慢，如何排查？",
         "我发现是 API 响应慢，怎么查？",
         "是数据库查询慢，SQL 怎么优化？"],
        ["网页加载慢的排查步骤：\n\n1. 打开 Chrome DevTools → Network 标签\n2. 刷新页面，查看瀑布图\n3. 重点关注：\n   - Time to First Byte (TTFB)：服务器响应速度\n   - 最大的资源文件（图片/JS/CSS）\n   - 阻塞渲染的资源\n\n常见原因：未压缩的图片、未打包的 JS、没有 CDN、服务器响应慢",
         "API 响应慢的排查：\n\n```python\n# 1. 加日志计时\nimport time, logging\n\n@app.middleware('http')\nasync def log_requests(request, call_next):\n    t = time.time()\n    response = await call_next(request)\n    duration = time.time() - t\n    logging.info(f'{request.url} took {duration*1000:.0f}ms')\n    return response\n\n# 2. 用 py-spy 做性能分析\n# pip install py-spy\n# py-spy top --pid <pid>\n\n# 3. 查看慢查询日志\n# PostgreSQL: log_min_duration_statement = 100  # 记录>100ms的查询\n```",
         "SQL 查询优化步骤：\n\n```sql\n-- 1. 用 EXPLAIN ANALYZE 查看执行计划\nEXPLAIN ANALYZE\nSELECT * FROM orders WHERE user_id = 100;\n\n-- 关注：Seq Scan（全表扫描）→ 需要加索引\n-- 好的情况：Index Scan\n\n-- 2. 添加索引\nCREATE INDEX CONCURRENTLY idx_orders_user_id ON orders(user_id);\n-- CONCURRENTLY 不锁表\n\n-- 3. 只查需要的列\n-- ❌ SELECT * FROM orders\n-- ✅ SELECT id, status, total FROM orders\n\n-- 4. 用 LIMIT 限制结果\nSELECT id, total FROM orders\nWHERE user_id = 100\nORDER BY created_at DESC\nLIMIT 20;\n```"]
    ),
    c(
        ["我想从前端转全栈开发，应该学什么？",
         "Node.js 和 Python 后端怎么选？",
         "有没有推荐的学习路线和时间表？"],
        ["前端转全栈需要补充的后端知识：\n\n**核心技能**\n1. 服务器语言（Node.js / Python / Go）\n2. HTTP 协议深入（状态码、Header、Cookie/Session）\n3. 数据库（SQL + NoSQL 各一种）\n4. API 设计（RESTful / GraphQL）\n5. 身份认证（JWT / OAuth）\n6. 部署运维（Linux 基础、Docker）\n\n前端最大的优势：已经懂 HTTP、JSON、异步编程，上手很快！",
         "对前端来说：\n\n**Node.js（推荐先学）**\n- 语言一致（JS/TS），学习曲线最低\n- 生态：Express / Fastify / NestJS\n- 适合：实时应用、BFF 层\n\n**Python（推荐作为第二语言）**\n- 语法简洁，代码量少\n- 生态：FastAPI（现代）/ Django（全功能）\n- 特别适合：AI/ML 集成\n\n**建议**：先用 Node.js + Express 写几个项目，熟悉后端思维；再学 Python/FastAPI，拓展 AI 方向",
         "6 个月全栈学习路线：\n\n**第 1 月：Node.js 基础**\n- 每天 1-2 小时，跟着做 TODO List API\n- 目标：能独立写增删改查接口\n\n**第 2 月：数据库**\n- PostgreSQL 基础 SQL\n- SQLAlchemy / Prisma ORM\n- 目标：会设计表结构，能做关联查询\n\n**第 3 月：认证与安全**\n- JWT 用户认证\n- 密码哈希、XSS/CSRF 防御\n\n**第 4 月：实战项目**\n- 独立完成一个有前后端的应用（博客/Todo/聊天室）\n\n**第 5-6 月：进阶**\n- Docker 部署\n- Redis 缓存\n- CI/CD 流程\n\n**关键**：每个阶段都要做项目，不要只看教程！"]
    ),
]
for item in deep_multi:
    DATA.append({"category": "多轮深度", "conversations": item})

# ─────────────────────────────────────────────
# 加载原有数据并合并
# ─────────────────────────────────────────────
with open("data/training/train_data.json", encoding="utf-8") as f:
    old_data = json.load(f)

# 合并并去重（按首条消息内容）
all_data = old_data + DATA
seen = set()
unique_data = []
for item in all_data:
    key = item["conversations"][0]["content"][:50]
    if key not in seen:
        seen.add(key)
        unique_data.append(item)

random.seed(42)
random.shuffle(unique_data)

# 统计
from collections import Counter
cats = Counter(x["category"] for x in unique_data)
total_chars = sum(len(m["content"]) for x in unique_data for m in x["conversations"])

print("="*55)
print(f"✅ 训练数据扩充完成")
print("="*55)
print(f"总对话数:  {len(unique_data)} 条")
print(f"总字符数:  {total_chars:,}")
print(f"平均字符:  {total_chars//len(unique_data)}")
print()
for cat, cnt in sorted(cats.items(), key=lambda x:-x[1]):
    print(f"  {cat}: {cnt} 条")

# 保存
out = "data/training/train_data.json"
with open(out, "w", encoding="utf-8") as f:
    json.dump(unique_data, f, ensure_ascii=False, indent=2)

print(f"\n📁 已保存: {out}")
print(f"📦 文件大小: {os.path.getsize(out)/1024:.0f} KB")
