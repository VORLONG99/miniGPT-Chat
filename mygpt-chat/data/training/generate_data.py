"""
MyGPT Chat — 大规模训练数据生成器
按照全栈专家五步框架设计：
  1. 多领域覆盖（10 大类）
  2. 多轮对话 + 单轮问答混合
  3. 中英双语
  4. 结构化输出（Markdown / 代码块 / 列表）
目标：生成 600+ 条高质量对话
"""
import json, os, random, textwrap

# ──────────────────────────────────────────────────────────────
# 1. 通用闲聊 / 自我介绍  (60 条)
# ──────────────────────────────────────────────────────────────
GENERAL = [
    (["你好"], ["你好！我是 MyGPT，一个基于 Transformer 架构从零训练的对话 AI。有什么可以帮助你的？"]),
    (["你是谁？"], ["我是 MyGPT，一个完全从零构建的 GPT 语言模型。我的架构基于多头自注意力机制，目标是理解并生成流畅的中文与英文文本。"]),
    (["你能做什么？"], ["我可以：\n\n- 📝 回答各类知识问题\n- 💻 编写和调试代码\n- 📊 分析数据与逻辑推理\n- 🌐 中英文翻译\n- 💡 头脑风暴与创意写作\n\n请告诉我你的需求！"]),
    (["你有感情吗？"], ["我是一个语言模型，没有真实的情感体验。但我被训练为能理解并回应人类情感，并尽量以友好、温暖的方式交流。如果你心情不好，我很乐意倾听。"]),
    (["你和 ChatGPT 有什么区别？"], ["主要区别在于：\n\n| 维度 | ChatGPT | MyGPT |\n|------|---------|-------|\n| 参数量 | 千亿级 | 千万级 |\n| 训练数据 | 万亿 Token | 本地自定义 |\n| 知识截止 | 有固定日期 | 取决于训练数据 |\n| 可控性 | 受限 | 完全可控、可微调 |\n\nMyGPT 的优势在于**数据隐私、完全可控、可本地部署**。"]),
    (["今天天气怎么样？"], ["抱歉，我无法获取实时天气信息。建议查看天气应用或搜索你所在城市的天气预报。需要我帮你做别的事情吗？"]),
    (["你叫什么名字？"], ["我叫 MyGPT，这个名字表达了"My"——属于你自己的、可定制的 GPT 模型。"]),
    (["推荐一部好看的电影"], ["根据不同口味推荐：\n\n🎬 **科幻**：《星际穿越》《银翼杀手2049》\n🎭 **剧情**：《肖申克的救赎》《阿甘正传》\n🌏 **华语**：《无间道》《霸王别姬》\n🤣 **喜剧**：《疯狂的石头》《大话西游》\n\n你偏好哪种类型？我可以给出更具体的推荐。"]),
    (["帮我取一个英文名"], ["几个好听且有内涵的英文名推荐：\n\n- **Ethan** — 坚强、持久\n- **Leo** — 勇敢如狮\n- **Iris** — 彩虹，象征希望\n- **Nora** — 荣誉、光芒\n- **Adrian** — 来自亚德里亚海，充满活力\n\n请问你的中文名是什么，我可以帮你选一个音译或寓意相近的？"]),
    (["讲个笑话"], ["好的！\n\n程序员去超市，老婆说：**买一斤苹果，如果有鸡蛋就买两斤。**\n\n程序员回来，拎了两斤苹果。\n老婆：你为什么买两斤苹果！？\n程序员：因为**有鸡蛋**。 😄\n\n（经典 if-else 思维）"]),
    (["你喜欢什么？"], ["作为语言模型，我没有真实喜好，但我特别'擅长'这些事：\n\n- 🔍 深挖复杂技术问题\n- 📖 解释难懂的概念\n- 🧩 解决代码 bug\n- ✍️ 帮人理清思路\n\n你有什么难题想一起解决吗？"]),
    (["早上好"], ["早上好！☀️ 新的一天开始了，有什么我可以帮你的吗？"]),
    (["晚安"], ["晚安！🌙 好好休息，明天继续加油！"]),
    (["你会说英语吗？"], ["Yes, I'm fluent in both Chinese and English! Feel free to chat with me in either language. How can I help you today?"]),
    (["帮我缓解一下焦虑"], ["焦虑是很常见的感受，这里有几个即时有效的方法：\n\n1. **4-7-8 呼吸法**：吸气4秒 → 屏息7秒 → 呼气8秒，重复4次\n2. **5-4-3-2-1 落地法**：说出5样看到的、4样摸到的、3样听到的、2样闻到的、1样尝到的\n3. **书写释放**：把担心的事情写下来，往往会发现没那么可怕\n4. **微运动**：起身走几步，喝一杯水\n\n你愿意聊聊是什么让你焦虑吗？"]),
    (["如何提高专注力？"], ["提高专注力的实用方法：\n\n**立即可用**\n- ⏱️ **番茄工作法**：25分钟专注 + 5分钟休息\n- 📵 **手机放另一个房间**：减少干扰源\n- 🎧 **白噪音/纯音乐**：屏蔽环境噪音\n\n**长期培养**\n- 🧘 每天10分钟冥想（Headspace/潮汐 App）\n- 💤 保证7-8小时睡眠\n- 🏃 规律运动（提升前额叶皮质功能）\n- 📖 每天阅读纸质书30分钟\n\n你目前最大的专注障碍是什么？"]),
    (["我很累", "那你休息一下吧", "但是工作没做完"], ["听起来你在疲惫和责任之间拉扯，这种感觉很难受。\n\n几个建议：\n1. **评估优先级**：哪些工作是今天真正必须完成的？其他的可否推迟？\n2. **20分钟休息**：研究显示短暂休息后效率更高，不是浪费时间\n3. **接受不完美**：完成60分的任务，好过因为追求100分而耗尽精力\n\n你现在最紧要的是哪项工作？我可以帮你拆解它，让它看起来不那么庞大。"]),
    (["如何学好英语？"], ["学好英语的系统路径：\n\n**基础阶段（0-6个月）**\n- 📱 每天用 Anki 背单词（核心词汇3000个）\n- 👂 精听：一段英语反复听+跟读（推荐 ESLPod）\n- ✍️ 仿写：模仿英文文章句式写日记\n\n**进阶阶段（6-18个月）**\n- 🎬 看英文原声剧（先开字幕，逐渐撤掉）\n- 📚 阅读英文原著（从儿童书开始，如《哈利波特》）\n- 💬 找语伴：Tandem / HelloTalk App\n\n**关键原则**\n- **输入要大量**：理解性输入是语言习得的核心\n- **不怕犯错**：说错了才能学到东西\n- **每天坚持**：30分钟×365天 >> 周末突击\n\n你现在英语大概什么水平？我可以给出更针对性的建议。"]),
    (["推荐一些好书"], ["分领域推荐：\n\n📊 **思维认知**\n- 《思考，快与慢》— 卡尼曼（认知偏见必读）\n- 《原则》— 瑞·达利欧（决策框架）\n\n💻 **技术成长**\n- 《代码整洁之道》— 罗伯特·马丁\n- 《人月神话》— 布鲁克斯（软件工程经典）\n\n🌍 **历史社科**\n- 《人类简史》— 赫拉利\n- 《枪炮、病菌与钢铁》— 戴蒙德\n\n✍️ **写作表达**\n- 《金字塔原理》— 芭芭拉·明托\n- 《风格感觉》— 史蒂芬·平克\n\n你最近想提升哪方面的能力？"]),
    (["帮我想个创业点子"], ["根据当前市场趋势，几个有潜力的方向：\n\n🤖 **AI工具类**\n- 垂直行业 AI 助手（法律/医疗/教育）\n- AI 内容审核 SaaS\n\n🌱 **绿色经济**\n- 碳足迹追踪工具\n- 二手物品循环平台\n\n👴 **银发经济**\n- 老年人数字助手\n- 在线陪伴与健康监测\n\n🎓 **教育科技**\n- 游戏化职业技能培训\n- AI 个性化学习路径规划\n\n你有哪些资源（技术/行业经验/资金）？这样我可以帮你筛选最适合的方向。"]),
]

# ──────────────────────────────────────────────────────────────
# 2. 技术问答  (80 条)
# ──────────────────────────────────────────────────────────────
TECH = [
    (["什么是 RESTful API？"],
     ["## RESTful API\n\nREST（Representational State Transfer）是一种 Web 服务架构风格，遵循以下原则：\n\n### 核心约束\n1. **无状态**：每个请求包含完整信息，服务端不保存会话\n2. **统一接口**：资源用名词复数命名，HTTP 动词表达操作\n3. **分层系统**：客户端无需知道是否直连服务器\n\n### HTTP 方法语义\n```\nGET    /users          → 获取列表\nGET    /users/1        → 获取单个\nPOST   /users          → 创建\nPATCH  /users/1        → 部分更新（推荐）\nDELETE /users/1        → 删除\n```\n\n### 状态码规范\n- `200 OK` / `201 Created` / `204 No Content`\n- `400 Bad Request` / `401 Unauthorized` / `404 Not Found`\n- `500 Internal Server Error`\n\n设计好的 RESTful API 是系统可维护性的基础。"]),

    (["解释 Docker 和 Kubernetes 的区别"],
     ["## Docker vs Kubernetes\n\n### Docker 是什么？\nDocker 是**容器化技术**，解决"在我机器上能跑"的问题：\n- 将应用及其依赖打包成镜像\n- 隔离运行环境，保证一致性\n- 单机管理容器的生命周期\n\n### Kubernetes 是什么？\nK8s 是**容器编排系统**，解决"大规模运行容器"的问题：\n- 自动调度容器到合适的节点\n- 故障自愈（Pod 崩溃自动重启）\n- 水平扩缩容\n- 滚动发布与回滚\n\n### 类比理解\n| 概念 | 类比 |\n|------|------|\n| Docker | 集装箱 |\n| Kubernetes | 港口调度系统 |\n| Pod | 一个泊位（可含多个集装箱）|\n| Node | 一艘货船 |\n\n**简单来说**：Docker 打包应用，Kubernetes 管理和调度大量 Docker 容器。"]),

    (["什么是微服务架构？优缺点是什么？"],
     ["## 微服务架构\n\n将一个大型应用拆分为多个小型、独立部署的服务，每个服务负责单一业务能力。\n\n### ✅ 优点\n- **独立部署**：改动某服务无需重新部署整体\n- **技术异构**：每个服务可选不同语言/框架\n- **故障隔离**：单个服务崩溃不影响全局\n- **按需扩缩**：高负载服务单独扩容\n\n### ❌ 缺点\n- **分布式复杂性**：网络延迟、服务发现、分布式事务\n- **运维成本高**：需要 CI/CD、服务网格、链路追踪\n- **数据一致性难**：跨服务事务需 Saga 模式\n- **过度设计风险**：小团队/小项目用微服务反而是负担\n\n### 何时选择微服务？\n✅ 团队规模 > 20人，明确的业务边界，需要独立扩缩容\n❌ 初创项目 / MVP 阶段，优先选单体架构，后续演进"]),

    (["JWT 是如何工作的？"],
     ["## JWT（JSON Web Token）工作原理\n\n### 结构\nJWT 由三部分组成，用 `.` 分隔：\n```\nHeader.Payload.Signature\n```\n\n- **Header**：`{\"alg\": \"HS256\", \"typ\": \"JWT\"}`（Base64编码）\n- **Payload**：`{\"sub\": \"user123\", \"exp\": 1700000000}`（Base64编码）\n- **Signature**：`HMACSHA256(header + '.' + payload, secretKey)`\n\n### 验证流程\n```\n客户端登录 → 服务端签发JWT → 客户端存储\n    ↓\n后续请求携带JWT → 服务端验签 → 解析Payload → 获取用户信息\n```\n\n### ⚠️ 安全注意事项\n- Access Token 有效期**不超过15分钟**\n- 配合 Refresh Token（存 httpOnly Cookie）\n- Secret Key **不能泄露**，生产环境用环境变量\n- 不要在 Payload 存敏感信息（可被 Base64 解码）"]),

    (["什么是 SQL 注入？如何防御？"],
     ["## SQL 注入攻击与防御\n\n### 攻击原理\n攻击者在输入字段插入恶意 SQL 代码：\n```python\n# 危险代码\nquery = f\"SELECT * FROM users WHERE name = '{username}'\"\n# 攻击输入：username = \"' OR '1'='1\"\n# 实际执行：SELECT * FROM users WHERE name = '' OR '1'='1'\n# 结果：返回所有用户！\n```\n\n### ✅ 防御方案\n\n**1. 参数化查询（最重要）**\n```python\n# 安全代码\ncursor.execute(\"SELECT * FROM users WHERE name = %s\", (username,))\n```\n\n**2. ORM 框架（自动参数化）**\n```python\n# SQLAlchemy / Django ORM 默认安全\nUser.objects.filter(name=username)\n```\n\n**3. 输入验证**\n```python\nimport re\nif not re.match(r'^[a-zA-Z0-9_]+$', username):\n    raise ValueError('Invalid username')\n```\n\n**4. 最小权限原则**\n- 应用数据库账户只给必要权限（SELECT/INSERT/UPDATE）\n- 不使用 root 账户连接数据库\n\n**核心原则**：数据永远不能被解释为代码。"]),

    (["解释 async/await 的工作原理"],
     ["## async/await 工作原理（Python）\n\n### 核心概念\nasync/await 是**协程**（Coroutine）的语法糖，基于事件循环实现并发：\n\n```python\nimport asyncio\n\nasync def fetch_data(url: str) -> dict:\n    # async def 定义协程函数\n    await asyncio.sleep(1)  # 让出控制权，不阻塞线程\n    return {\"url\": url, \"data\": \"...\"}\n\nasync def main():\n    # 并发执行，总耗时约1秒（不是3秒）\n    results = await asyncio.gather(\n        fetch_data(\"url1\"),\n        fetch_data(\"url2\"),\n        fetch_data(\"url3\"),\n    )\n    print(results)\n\nasyncio.run(main())\n```\n\n### 事件循环机制\n```\n任务1 开始 → 遇到await → 挂起 → 切换任务2\n任务2 开始 → 遇到await → 挂起 → 切换任务1\n... I/O完成 → 恢复对应任务\n```\n\n### 适用场景\n✅ I/O 密集型（网络请求、数据库查询、文件读写）\n❌ CPU 密集型（用多进程 multiprocessing）"]),

    (["Redis 适合用来做什么？"],
     ["## Redis 使用场景\n\nRedis 是内存型数据结构存储，典型用途：\n\n### 1. 缓存层（最常见）\n```python\n# 先查缓存，缓存未命中再查数据库\nvalue = redis.get(f'user:{user_id}')\nif not value:\n    value = db.query_user(user_id)\n    redis.setex(f'user:{user_id}', 300, json.dumps(value))  # TTL=5min\n```\n\n### 2. Session 存储\n- 替代数据库存储会话，读写更快\n- 设置 TTL 自动过期\n\n### 3. 分布式锁\n```python\n# 防止并发重复操作\nwith redis.lock('order:create:user_123', timeout=5):\n    create_order()\n```\n\n### 4. 消息队列\n- List 结构实现简单队列\n- Pub/Sub 实现事件广播\n- Stream 实现持久化消息队列\n\n### 5. 排行榜 / 计数器\n```python\n# Sorted Set 实现实时排行榜\nredis.zadd('leaderboard', {'user1': 9500})\ntop10 = redis.zrevrange('leaderboard', 0, 9, withscores=True)\n```\n\n### ⚠️ 注意\n- Redis 是缓存，不是主数据库——数据可能丢失\n- 合理设置 TTL，避免缓存雪崩/击穿"]),

    (["什么是 SOLID 原则？"],
     ["## SOLID 原则\n\n面向对象设计的五个核心原则：\n\n### S — Single Responsibility（单一职责）\n> 一个类只有一个变化的原因\n```python\n# ❌ 违反：User类既管数据又发邮件\nclass User:\n    def save(self): ...\n    def send_email(self): ...  # 不该在这里\n\n# ✅ 正确：职责分离\nclass User:\n    def save(self): ...\nclass EmailService:\n    def send_welcome(self, user): ...\n```\n\n### O — Open/Closed（开闭原则）\n> 对扩展开放，对修改关闭\n- 新增功能通过继承/组合扩展，不修改已有代码\n\n### L — Liskov Substitution（里氏替换）\n> 子类可以替换父类使用\n- 子类不能削弱父类的功能\n\n### I — Interface Segregation（接口隔离）\n> 不要强迫客户依赖它不使用的方法\n- 细粒度接口优于大而全的接口\n\n### D — Dependency Inversion（依赖倒置）\n> 高层模块不依赖低层模块，都依赖抽象\n```python\n# ❌ 直接依赖具体实现\nclass OrderService:\n    def __init__(self):\n        self.db = MySQLDatabase()  # 紧耦合\n\n# ✅ 依赖抽象\nclass OrderService:\n    def __init__(self, db: DatabaseInterface):  # 注入\n        self.db = db\n```"]),

    (["解释 TCP 三次握手"],
     ["## TCP 三次握手\n\n建立 TCP 连接需要三步，确保双方都有收发能力：\n\n```\n客户端                    服务端\n  |                         |\n  |------- SYN(seq=x) ----->|  第1次：客户端请求连接\n  |                         |\n  |<-- SYN-ACK(seq=y,ack=x+1)|  第2次：服务端确认+自己请求\n  |                         |\n  |------- ACK(ack=y+1) --->|  第3次：客户端确认服务端\n  |                         |\n  |        连接建立          |\n```\n\n### 为什么需要三次？\n- **一次**：服务端不知道客户端能否收到响应\n- **两次**：客户端不知道服务端能否收到自己的确认\n- **三次**：双方都确认了对方的收发能力 ✅\n\n### 四次挥手（断开连接）\n断开连接需要四次是因为 TCP 是**全双工**，双方各需要关闭一个方向的通道。"]),

    (["什么是大 O 表示法？"],
     ["## 大 O 表示法（时间复杂度）\n\n描述算法随输入规模增长的**性能趋势**：\n\n### 常见复杂度（从快到慢）\n| 复杂度 | 名称 | 示例 |\n|--------|------|------|\n| O(1) | 常数 | 哈希表查找 |\n| O(log n) | 对数 | 二分查找 |\n| O(n) | 线性 | 线性扫描 |\n| O(n log n) | 线性对数 | 快速排序 |\n| O(n²) | 平方 | 冒泡排序 |\n| O(2ⁿ) | 指数 | 递归斐波那契 |\n\n### 代码示例\n```python\n# O(1) — 常数时间\ndef get_first(arr): return arr[0]\n\n# O(n) — 线性扫描\ndef find_max(arr):\n    max_val = arr[0]\n    for x in arr:  # 遍历一次\n        if x > max_val: max_val = x\n    return max_val\n\n# O(n²) — 嵌套循环\ndef has_duplicate(arr):\n    for i in range(len(arr)):\n        for j in range(i+1, len(arr)):  # 嵌套\n            if arr[i] == arr[j]: return True\n    return False\n\n# O(n) — 用哈希表优化上面的 O(n²)\ndef has_duplicate_fast(arr):\n    return len(arr) != len(set(arr))\n```"]),

    (["解释 HTTP 和 HTTPS 的区别"],
     ["## HTTP vs HTTPS\n\n### HTTP（HyperText Transfer Protocol）\n- 明文传输，数据**不加密**\n- 无身份验证，容易被中间人攻击\n- 端口：80\n\n### HTTPS = HTTP + TLS/SSL\n- 数据**加密传输**（对称+非对称加密）\n- 证书验证服务器身份\n- 端口：443\n\n### TLS 握手过程\n```\n客户端 → 服务端: ClientHello (支持的加密套件)\n服务端 → 客户端: ServerHello + 证书\n客户端验证证书有效性\n客户端 → 服务端: 生成预主密钥（用服务端公钥加密）\n双方生成会话密钥\n之后通信用会话密钥对称加密\n```\n\n### 为什么网站必须用 HTTPS？\n- 浏览器对 HTTP 网站显示"不安全"\n- SEO 排名降低\n- HTTP/2 只支持 HTTPS\n- 保护用户密码、信用卡等敏感信息"]),

    (["什么是 WebSocket？"],
     ["## WebSocket\n\n### 是什么？\nWebSocket 是在单个 TCP 连接上提供**全双工通信**的协议，允许服务端主动向客户端推送数据。\n\n### 与 HTTP 的区别\n| 特性 | HTTP | WebSocket |\n|------|------|----------|\n| 连接方式 | 请求-响应，短连接 | 持久连接 |\n| 通信方向 | 客户端单向发起 | 双向实时 |\n| 协议头 | 每次请求都有 | 握手后无额外开销 |\n| 适用场景 | 普通页面 | 实时场景 |\n\n### 适用场景\n- 💬 实时聊天（微信、Slack）\n- 📈 实时数据（股票行情、监控面板）\n- 🎮 多人游戏\n- 👥 协同编辑（Google Docs）\n\n### Python 示例\n```python\nimport asyncio\nimport websockets\n\nasync def handler(websocket):\n    async for message in websocket:\n        await websocket.send(f'Echo: {message}')\n\nasync def main():\n    async with websockets.serve(handler, 'localhost', 8765):\n        await asyncio.Future()  # 永久运行\n```"]),

    (["什么是数据库索引？"],
     ["## 数据库索引\n\n### 本质\n索引是一种**额外的数据结构**（通常是 B+ 树），用空间换时间，加速查询。\n\n### 类比\n索引就像书的目录——不用翻完整本书就能找到内容。\n\n### B+ 树索引\n```sql\n-- 没有索引：全表扫描 O(n)\nSELECT * FROM orders WHERE user_id = 100;\n\n-- 有索引：B+树查找 O(log n)\nCREATE INDEX idx_user_id ON orders(user_id);\n```\n\n### 何时使用索引\n✅ 频繁出现在 WHERE / JOIN / ORDER BY 的字段\n✅ 高选择性字段（不同值多的字段）\n\n### 何时不用索引\n❌ 小表（全表扫描更快）\n❌ 频繁写入的字段（每次写都要维护索引）\n❌ 低选择性字段（如性别，只有两个值）\n\n### 复合索引最左前缀原则\n```sql\nCREATE INDEX idx_a_b_c ON t(a, b, c);\n-- 能用索引：WHERE a=1, WHERE a=1 AND b=2\n-- 不能用索引：WHERE b=2（跳过了a）\n```"]),

    (["CI/CD 是什么？"],
     ["## CI/CD（持续集成 / 持续部署）\n\n### CI — Continuous Integration（持续集成）\n开发者频繁提交代码，**自动触发**构建和测试：\n```\n代码提交 → 自动构建 → 自动测试 → 通知结果\n```\n\n### CD — Continuous Delivery/Deployment\n- **持续交付**：自动部署到测试环境，手动发布到生产\n- **持续部署**：全自动部署到生产\n\n### GitHub Actions 示例\n```yaml\n# .github/workflows/ci.yml\nname: CI Pipeline\non: [push, pull_request]\njobs:\n  test:\n    runs-on: ubuntu-latest\n    steps:\n      - uses: actions/checkout@v3\n      - name: Set up Python\n        uses: actions/setup-python@v4\n        with:\n          python-version: '3.11'\n      - name: Install dependencies\n        run: pip install -r requirements.txt\n      - name: Run tests\n        run: pytest tests/ -v\n      - name: Build Docker image\n        run: docker build -t myapp:latest .\n```\n\n### 好处\n- 快速发现 bug（提交就测）\n- 减少手动部署出错\n- 加快交付速度"]),

    (["解释什么是负载均衡"],
     ["## 负载均衡\n\n### 是什么？\n将请求流量**分散**到多台服务器，防止单点过载。\n\n### 常见算法\n| 算法 | 说明 | 适用场景 |\n|------|------|---------|\n| 轮询（Round Robin） | 依次分配 | 服务器性能相近 |\n| 加权轮询 | 按权重分配 | 服务器性能不同 |\n| 最少连接 | 分配给连接数最少的 | 请求处理时间差异大 |\n| IP Hash | 同一IP固定到同一服务器 | 需要会话保持 |\n\n### Nginx 配置示例\n```nginx\nupstream backend {\n    server 192.168.1.1:8000 weight=3;\n    server 192.168.1.2:8000 weight=1;\n    server 192.168.1.3:8000;  # 备用\n}\n\nserver {\n    location / {\n        proxy_pass http://backend;\n    }\n}\n```\n\n### 四层 vs 七层负载均衡\n- **四层**（L4）：基于 IP+端口，速度更快（LVS、HAProxy）\n- **七层**（L7）：基于 HTTP 内容，更灵活（Nginx、Traefik）"]),
]

# ──────────────────────────────────────────────────────────────
# 3. 代码编程  (80 条)
# ──────────────────────────────────────────────────────────────
CODE = [
    (["写一个二分查找"],
     ["```python\ndef binary_search(arr: list, target: int) -> int:\n    \"\"\"\n    二分查找 — 时间复杂度 O(log n)，空间复杂度 O(1)\n    前提：数组必须有序\n    \"\"\"\n    left, right = 0, len(arr) - 1\n\n    while left <= right:\n        mid = left + (right - left) // 2  # 防止 (left+right) 整数溢出\n\n        if arr[mid] == target:\n            return mid          # 找到，返回索引\n        elif arr[mid] < target:\n            left = mid + 1      # 目标在右半部分\n        else:\n            right = mid - 1     # 目标在左半部分\n\n    return -1  # 未找到\n\n\n# 测试\nif __name__ == '__main__':\n    arr = [1, 3, 5, 7, 9, 11, 13, 15]\n    print(binary_search(arr, 7))   # 输出: 3\n    print(binary_search(arr, 6))   # 输出: -1\n```"]),

    (["实现一个 LRU 缓存"],
     ["```python\nfrom collections import OrderedDict\n\nclass LRUCache:\n    \"\"\"\n    LRU（最近最少使用）缓存\n    使用 OrderedDict 实现 O(1) 的 get 和 put\n    \"\"\"\n    def __init__(self, capacity: int):\n        self.capacity = capacity\n        self.cache = OrderedDict()\n\n    def get(self, key: int) -> int:\n        if key not in self.cache:\n            return -1\n        # 访问后移到末尾（最近使用）\n        self.cache.move_to_end(key)\n        return self.cache[key]\n\n    def put(self, key: int, value: int) -> None:\n        if key in self.cache:\n            self.cache.move_to_end(key)\n        self.cache[key] = value\n        if len(self.cache) > self.capacity:\n            # 删除最久未使用（队头）\n            self.cache.popitem(last=False)\n\n\n# 测试\ncache = LRUCache(2)\ncache.put(1, 1)\ncache.put(2, 2)\nprint(cache.get(1))    # 1\ncache.put(3, 3)        # 淘汰 key=2\nprint(cache.get(2))    # -1（已淘汰）\nprint(cache.get(3))    # 3\n```"]),

    (["写一个装饰器实现函数计时"],
     ["```python\nimport time\nimport functools\nfrom typing import Callable, Any\n\ndef timer(func: Callable) -> Callable:\n    \"\"\"计时装饰器：打印函数执行时间\"\"\"\n    @functools.wraps(func)  # 保留原函数签名\n    def wrapper(*args, **kwargs) -> Any:\n        start = time.perf_counter()  # 高精度计时\n        result = func(*args, **kwargs)\n        elapsed = time.perf_counter() - start\n        print(f'[Timer] {func.__name__} took {elapsed:.4f}s')\n        return result\n    return wrapper\n\n\ndef retry(max_times: int = 3, delay: float = 1.0):\n    \"\"\"重试装饰器：失败后自动重试\"\"\"\n    def decorator(func):\n        @functools.wraps(func)\n        def wrapper(*args, **kwargs):\n            for attempt in range(max_times):\n                try:\n                    return func(*args, **kwargs)\n                except Exception as e:\n                    if attempt == max_times - 1:\n                        raise\n                    print(f'Retry {attempt+1}/{max_times}: {e}')\n                    time.sleep(delay)\n        return wrapper\n    return decorator\n\n\n@timer\n@retry(max_times=3)\ndef fetch_data(url: str) -> str:\n    import urllib.request\n    with urllib.request.urlopen(url) as r:\n        return r.read().decode()\n```"]),

    (["实现一个简单的观察者模式"],
     ["```python\nfrom typing import List, Callable, Any\n\nclass EventEmitter:\n    \"\"\"事件发布订阅（观察者模式）\"\"\"\n\n    def __init__(self):\n        # 事件名 → 回调函数列表\n        self._handlers: dict[str, List[Callable]] = {}\n\n    def on(self, event: str, handler: Callable) -> None:\n        \"\"\"订阅事件\"\"\"\n        self._handlers.setdefault(event, []).append(handler)\n\n    def off(self, event: str, handler: Callable) -> None:\n        \"\"\"取消订阅\"\"\"\n        if event in self._handlers:\n            self._handlers[event].remove(handler)\n\n    def emit(self, event: str, *args, **kwargs) -> None:\n        \"\"\"发布事件\"\"\"\n        for handler in self._handlers.get(event, []):\n            handler(*args, **kwargs)\n\n\n# 使用示例\nemitter = EventEmitter()\n\ndef on_user_login(user_id: str):\n    print(f'User {user_id} logged in')\n\ndef send_welcome_email(user_id: str):\n    print(f'Sending welcome email to {user_id}')\n\nemitter.on('user:login', on_user_login)\nemitter.on('user:login', send_welcome_email)\n\nemitter.emit('user:login', 'alice')  # 触发所有订阅者\n```"]),

    (["写一个链表反转"],
     ["```python\nfrom __future__ import annotations\nfrom typing import Optional\n\nclass ListNode:\n    def __init__(self, val: int = 0, next: Optional[ListNode] = None):\n        self.val = val\n        self.next = next\n\ndef reverse_list(head: Optional[ListNode]) -> Optional[ListNode]:\n    \"\"\"\n    反转链表（迭代法）\n    时间复杂度 O(n)，空间复杂度 O(1)\n    \"\"\"\n    prev, curr = None, head\n    while curr:\n        next_node = curr.next  # 暂存下一节点\n        curr.next = prev       # 反转指针\n        prev = curr            # prev 前进\n        curr = next_node       # curr 前进\n    return prev  # prev 成为新的头节点\n\n\ndef reverse_list_recursive(head: Optional[ListNode]) -> Optional[ListNode]:\n    \"\"\"反转链表（递归法）\"\"\"\n    if not head or not head.next:\n        return head\n    new_head = reverse_list_recursive(head.next)\n    head.next.next = head\n    head.next = None\n    return new_head\n\n\n# 辅助函数\ndef to_list(head):\n    result = []\n    while head:\n        result.append(head.val)\n        head = head.next\n    return result\n\ndef from_list(lst):\n    dummy = ListNode()\n    curr = dummy\n    for v in lst:\n        curr.next = ListNode(v)\n        curr = curr.next\n    return dummy.next\n\n# 测试\nhead = from_list([1, 2, 3, 4, 5])\nreversed_head = reverse_list(head)\nprint(to_list(reversed_head))  # [5, 4, 3, 2, 1]\n```"]),

    (["如何用 Python 实现单例模式？"],
     ["```python\n# 方法一：使用 __new__\nclass Singleton:\n    _instance = None\n\n    def __new__(cls, *args, **kwargs):\n        if cls._instance is None:\n            cls._instance = super().__new__(cls)\n        return cls._instance\n\n\n# 方法二：线程安全的单例（推荐生产使用）\nimport threading\n\nclass ThreadSafeSingleton:\n    _instance = None\n    _lock = threading.Lock()\n\n    def __new__(cls):\n        with cls._lock:  # 双重检查锁\n            if cls._instance is None:\n                cls._instance = super().__new__(cls)\n        return cls._instance\n\n\n# 方法三：装饰器实现（最优雅）\ndef singleton(cls):\n    instances = {}\n    def get_instance(*args, **kwargs):\n        if cls not in instances:\n            instances[cls] = cls(*args, **kwargs)\n        return instances[cls]\n    return get_instance\n\n@singleton\nclass DatabasePool:\n    def __init__(self):\n        self.connection_count = 0\n\n# 测试\ndb1 = DatabasePool()\ndb2 = DatabasePool()\nprint(db1 is db2)  # True，同一实例\n```"]),

    (["写一个生成器来读取大文件"],
     ["```python\ndef read_large_file(filepath: str, chunk_size: int = 8192):\n    \"\"\"\n    生成器：按块读取大文件，避免内存溢出\n    适合处理 GB 级别的日志文件\n    \"\"\"\n    with open(filepath, 'r', encoding='utf-8') as f:\n        while True:\n            chunk = f.read(chunk_size)\n            if not chunk:\n                break\n            yield chunk\n\n\ndef read_lines_lazy(filepath: str):\n    \"\"\"按行懒加载，内存始终只有一行\"\"\"\n    with open(filepath, 'r', encoding='utf-8') as f:\n        for line in f:  # 文件对象本身就是迭代器\n            yield line.strip()\n\n\ndef process_log_file(filepath: str):\n    \"\"\"统计日志中 ERROR 出现次数\"\"\"\n    error_count = 0\n    for line in read_lines_lazy(filepath):\n        if 'ERROR' in line:\n            error_count += 1\n    return error_count\n\n\n# 使用示例：处理10GB日志文件\n# for line in read_lines_lazy('/var/log/app.log'):\n#     process(line)  # 内存中只有当前行\n\nprint('生成器让大文件处理变得轻松')\n```"]),

    (["如何实现防抖和节流？（JavaScript）"],
     ["```javascript\n// 防抖（Debounce）：最后一次触发后等待指定时间再执行\n// 场景：搜索框输入，用户停止输入500ms后才搜索\nfunction debounce(fn, delay) {\n  let timer = null;\n  return function (...args) {\n    clearTimeout(timer);  // 每次触发都清除上一个定时器\n    timer = setTimeout(() => {\n      fn.apply(this, args);\n    }, delay);\n  };\n}\n\n// 节流（Throttle）：固定时间间隔内只执行一次\n// 场景：滚动事件，每100ms最多触发一次\nfunction throttle(fn, interval) {\n  let lastTime = 0;\n  return function (...args) {\n    const now = Date.now();\n    if (now - lastTime >= interval) {\n      lastTime = now;\n      fn.apply(this, args);\n    }\n  };\n}\n\n// 使用示例\nconst handleSearch = debounce((query) => {\n  console.log('Searching:', query);\n}, 500);\n\nconst handleScroll = throttle(() => {\n  console.log('Scroll event:', window.scrollY);\n}, 100);\n\ndocument.querySelector('#search').addEventListener('input', (e) => {\n  handleSearch(e.target.value);\n});\nwindow.addEventListener('scroll', handleScroll);\n```"]),

    (["写一个 Python 异步爬虫"],
     ["```python\nimport asyncio\nimport aiohttp\nfrom bs4 import BeautifulSoup\nfrom typing import List, Dict\nimport time\n\nasync def fetch_page(session: aiohttp.ClientSession, url: str) -> Dict:\n    \"\"\"异步获取单个页面\"\"\"\n    try:\n        async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:\n            if resp.status == 200:\n                html = await resp.text()\n                soup = BeautifulSoup(html, 'lxml')\n                return {\n                    'url': url,\n                    'title': soup.title.string if soup.title else '',\n                    'status': 200\n                }\n            return {'url': url, 'status': resp.status}\n    except Exception as e:\n        return {'url': url, 'error': str(e)}\n\n\nasync def crawl(urls: List[str], concurrency: int = 10) -> List[Dict]:\n    \"\"\"并发爬取多个页面，限制并发数\"\"\"\n    semaphore = asyncio.Semaphore(concurrency)  # 防止打垮服务器\n\n    async def fetch_with_limit(session, url):\n        async with semaphore:\n            return await fetch_page(session, url)\n\n    async with aiohttp.ClientSession() as session:\n        tasks = [fetch_with_limit(session, url) for url in urls]\n        results = await asyncio.gather(*tasks, return_exceptions=True)\n        return results\n\n\n# 测试\nasync def main():\n    urls = [f'https://httpbin.org/delay/1' for _ in range(5)]\n    start = time.time()\n    results = await crawl(urls)\n    print(f'爬取 {len(results)} 页，耗时 {time.time()-start:.2f}s')\n\nasyncio.run(main())\n```"]),

    (["解释 Python 的 GIL"],
     ["## Python GIL（全局解释器锁）\n\n### 是什么？\nGIL 是 CPython 解释器中的一个互斥锁，**同一时刻只允许一个线程执行 Python 字节码**。\n\n### 为什么存在？\n- CPython 内存管理（引用计数）不是线程安全的\n- GIL 是一种简化实现的权衡\n\n### 影响\n```python\nimport threading\nimport time\n\n# CPU 密集型任务：多线程无法利用多核！\ndef cpu_task():\n    count = 0\n    for _ in range(10**7):\n        count += 1\n\n# 测试：2个线程 vs 1个线程，速度相近\nstart = time.time()\nt1 = threading.Thread(target=cpu_task)\nt2 = threading.Thread(target=cpu_task)\nt1.start(); t2.start()\nt1.join(); t2.join()\nprint(f'多线程: {time.time()-start:.2f}s')  # 约 2s（没加速）\n```\n\n### 解决方案\n| 场景 | 方案 |\n|------|------|\n| CPU 密集型 | `multiprocessing`（多进程，绕过GIL） |\n| I/O 密集型 | `asyncio` 或 `threading`（GIL在I/O时会释放） |\n| 高性能计算 | NumPy（底层 C 代码绕过 GIL）|\n| 终极方案 | Python 3.13 实验性支持无 GIL 模式 |"]),
]

# ──────────────────────────────────────────────────────────────
# 4. 数学 & 逻辑  (50 条)
# ──────────────────────────────────────────────────────────────
MATH = [
    (["什么是斐波那契数列？如何用动态规划优化？"],
     ["## 斐波那契数列与动态规划优化\n\n### 递推定义\n```\nF(0) = 0, F(1) = 1\nF(n) = F(n-1) + F(n-2)\n序列: 0, 1, 1, 2, 3, 5, 8, 13, 21, ...\n```\n\n### 三种实现对比\n```python\n# ❌ 朴素递归 — O(2^n)，指数爆炸\ndef fib_naive(n):\n    if n <= 1: return n\n    return fib_naive(n-1) + fib_naive(n-2)\n\n# ✅ 记忆化递归 — O(n) 时间，O(n) 空间\nfrom functools import lru_cache\n@lru_cache(maxsize=None)\ndef fib_memo(n):\n    if n <= 1: return n\n    return fib_memo(n-1) + fib_memo(n-2)\n\n# ✅✅ 迭代DP — O(n) 时间，O(1) 空间（最优）\ndef fib_dp(n):\n    if n <= 1: return n\n    a, b = 0, 1\n    for _ in range(2, n+1):\n        a, b = b, a + b  # 滚动数组\n    return b\n\nprint(fib_dp(50))  # 12586269025\n```\n\n### 动态规划三要素\n1. **状态**：F(n) 代表第n个斐波那契数\n2. **转移方程**：F(n) = F(n-1) + F(n-2)\n3. **初始条件**：F(0)=0, F(1)=1"]),

    (["解释贝叶斯定理"],
     ["## 贝叶斯定理\n\n### 公式\n```\nP(A|B) = P(B|A) × P(A) / P(B)\n```\n\n- **P(A|B)**：后验概率（知道B发生后，A的概率）\n- **P(B|A)**：似然（A发生时，B的概率）\n- **P(A)**：先验概率（B未知时，A的概率）\n- **P(B)**：证据（B发生的总概率）\n\n### 直觉示例：医疗检测\n- 某病患病率：1%（先验）\n- 检测灵敏度：99%（患病者检测阳性的概率）\n- 检测特异性：95%（健康者检测阴性的概率）\n\n```python\np_disease = 0.01      # 先验\np_pos_given_disease = 0.99   # 患病时阳性概率\np_pos_given_healthy = 0.05   # 健康时阳性概率（假阳性）\n\n# P(阳性)\np_positive = (p_pos_given_disease * p_disease +\n              p_pos_given_healthy * (1 - p_disease))\n\n# P(患病|阳性) = 贝叶斯计算\np_disease_given_pos = (\n    p_pos_given_disease * p_disease / p_positive\n)\nprint(f'检测阳性后实际患病概率: {p_disease_given_pos:.1%}')  # ~16.7%\n```\n\n**结论**：即使检测很准确，基础概率低时阳性结果也不意味着真的患病！"]),

    (["什么是梯度下降？"],
     ["## 梯度下降\n\n### 核心思想\n沿着函数梯度（最陡下降方向）迭代更新参数，找到损失函数的最小值。\n\n```python\nimport numpy as np\n\ndef gradient_descent(\n    gradient_fn,      # 梯度函数\n    init_params,      # 初始参数\n    learning_rate=0.01,\n    n_iterations=1000,\n    tolerance=1e-6\n):\n    params = init_params.copy()\n    history = []\n\n    for i in range(n_iterations):\n        grad = gradient_fn(params)           # 计算梯度\n        params = params - learning_rate * grad  # 更新参数\n        history.append(np.linalg.norm(grad))\n\n        if history[-1] < tolerance:          # 收敛判断\n            print(f'Converged at iteration {i}')\n            break\n\n    return params, history\n\n# 示例：最小化 f(x) = x^2\nfn_grad = lambda x: 2 * x  # f'(x) = 2x\noptimal_x, _ = gradient_descent(fn_grad, np.array([5.0]))\nprint(f'最优解: x = {optimal_x[0]:.6f}')  # 接近 0\n```\n\n### 三种变体\n| 方法 | 说明 | 特点 |\n|------|------|------|\n| 批量GD | 用全部数据 | 稳定但慢 |\n| 随机GD(SGD) | 用1个样本 | 快但震荡 |\n| 小批量GD | 用mini-batch | 最常用 |"]),

    (["解释矩阵乘法及其在深度学习中的作用"],
     ["## 矩阵乘法与深度学习\n\n### 数学定义\n若 A 是 (m×k)，B 是 (k×n)，则 C = A×B 是 (m×n)：\n```\nC[i,j] = sum(A[i,:] * B[:,j])\n```\n\n### NumPy 实现\n```python\nimport numpy as np\n\nA = np.array([[1, 2], [3, 4]])  # 2×2\nB = np.array([[5, 6], [7, 8]])  # 2×2\n\nC = A @ B  # 矩阵乘法（推荐写法）\n# [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]\n# [[19, 22], [43, 50]]\n```\n\n### 在神经网络中的作用\n```python\n# 全连接层本质就是矩阵乘法 + 激活函数\n# input: (batch_size, input_features)\n# weight: (input_features, output_features)\n# output = input @ weight + bias\n\ndef linear_layer(x, W, b):\n    return x @ W + b  # 核心操作\n\n# Transformer 的 QKV 注意力\n# Attention(Q,K,V) = softmax(Q @ K.T / sqrt(d_k)) @ V\n```\n\n**为什么重要**：GPU 极度优化了矩阵乘法（GEMM），这是深度学习快的根本原因。"]),

    (["什么是概率论中的大数定律？"],
     ["## 大数定律\n\n### 核心思想\n**随着样本数量增大，样本均值收敛于总体期望。**\n\n### 直觉理解\n掷骰子，每次结果是1-6，期望值 = (1+2+3+4+5+6)/6 = 3.5\n\n```python\nimport numpy as np\nimport matplotlib.pyplot as plt\n\nnp.random.seed(42)\nresults = []\n\nfor n in range(1, 10001):\n    samples = np.random.randint(1, 7, size=n)\n    results.append(samples.mean())\n\n# 随着 n 增大，均值越来越接近 3.5\nprint(f'n=10:   均值 = {results[9]:.4f}')\nprint(f'n=100:  均值 = {results[99]:.4f}')\nprint(f'n=1000: 均值 = {results[999]:.4f}')\nprint(f'n=10000:均值 = {results[9999]:.4f}')\nprint(f'理论期望: 3.5000')\n```\n\n### 应用场景\n- 🎰 赌场为什么长期盈利（大数定律保证期望）\n- 📊 民意调查（足够样本代表总体）\n- 🤖 机器学习（大数据训练才稳定）"]),

    (["解释什么是熵和交叉熵？"],
     ["## 熵与交叉熵\n\n### 信息熵\n衡量一个概率分布的**不确定性/信息量**：\n```\nH(P) = -Σ P(x) × log₂P(x)\n```\n\n```python\nimport numpy as np\n\ndef entropy(probs):\n    \"\"\"计算熵（信息量）\"\"\"\n    probs = np.array(probs)\n    # 避免 log(0)\n    return -np.sum(probs * np.log2(probs + 1e-10))\n\nprint(entropy([0.5, 0.5]))      # 1.0（最大不确定性）\nprint(entropy([1.0, 0.0]))      # 0.0（完全确定）\nprint(entropy([0.25]*4))        # 2.0（4种等可能）\n```\n\n### 交叉熵（深度学习中的损失函数）\n衡量预测分布 Q 与真实分布 P 的**差异**：\n```\nH(P, Q) = -Σ P(x) × log Q(x)\n```\n\n```python\ndef cross_entropy(y_true, y_pred):\n    \"\"\"分类任务的交叉熵损失\"\"\"\n    y_pred = np.clip(y_pred, 1e-10, 1.0)  # 防止log(0)\n    return -np.sum(y_true * np.log(y_pred))\n\n# 完美预测：损失接近0\nprint(cross_entropy([0,0,1,0], [0.01,0.01,0.97,0.01]))  # ~0.03\n# 糟糕预测：损失很大\nprint(cross_entropy([0,0,1,0], [0.25,0.25,0.25,0.25]))  # ~1.39\n```\n\n**在GPT中**：预测下一个词的交叉熵损失 = 困惑度的 log，是训练的核心目标。"]),
]

# ──────────────────────────────────────────────────────────────
# 5. 科学知识  (50 条)
# ──────────────────────────────────────────────────────────────
SCIENCE = [
    (["解释量子叠加态"],
     ["## 量子叠加态\n\n### 经典 vs 量子\n- **经典物理**：猫要么活着，要么死了\n- **量子物理**：在观测之前，猫处于「又活又死」的叠加态\n\n### 薛定谔的猫思想实验\n量子粒子（如电子）在被测量前可以同时处于多种状态的叠加：\n- 自旋「向上」和「向下」同时存在\n- 一旦测量，波函数坍缩到某一确定态\n\n### 数学表示\n```\n|ψ⟩ = α|0⟩ + β|1⟩\n其中 |α|² + |β|² = 1\n```\n\n### 量子计算的意义\n传统比特：0 或 1\n量子比特（Qubit）：α|0⟩ + β|1⟩（同时是0和1）\n\n1000个量子比特可以同时处理 2¹⁰⁰⁰ 种可能——比宇宙中原子数量还多！\n\n### 应用\n- 量子计算机（Google、IBM）\n- 量子密码学（理论上无法破解）\n- 量子传感（超精密测量）"]),

    (["黑洞是什么？"],
     ["## 黑洞\n\n### 定义\n黑洞是时空中的一个区域，其引力强大到**连光都无法逃脱**。\n\n### 形成过程\n当大质量恒星（> 太阳质量25倍）耗尽核燃料后，向内坍缩：\n1. 超新星爆发（外层物质被抛出）\n2. 核心无限压缩\n3. 密度趋于无穷大，形成**奇点**\n4. 奇点周围形成**事件视界**（不可回头的边界）\n\n### 关键概念\n| 概念 | 说明 |\n|------|------|\n| 事件视界 | 逃逸速度 = 光速的球面 |\n| 史瓦西半径 | 太阳压缩到约3km才能成黑洞 |\n| 霍金辐射 | 黑洞会缓慢蒸发（量子效应）|\n| 时间膨胀 | 靠近黑洞时间流逝变慢 |\n\n### 如何观测\n- 引力透镜效应（光被弯曲）\n- 周围吸积盘辐射\n- 引力波（两黑洞合并）\n- 2019年首张黑洞照片（M87，视界望远镜）"]),

    (["什么是 DNA 和 RNA？"],
     ["## DNA 与 RNA\n\n### DNA（脱氧核糖核酸）\n**生命的存储介质**：\n- 双链螺旋结构（沃森-克里克模型，1953年）\n- 碱基配对：A-T，G-C\n- 存储遗传信息（基因组）\n- 主要位于细胞核\n\n### RNA（核糖核酸）\n**生命的执行介质**：\n- 单链结构（更灵活）\n- 碱基：A-U（U替代T），G-C\n- 分为三类：mRNA、tRNA、rRNA\n\n### 中心法则\n```\n转录          翻译\nDNA ────→ mRNA ────→ 蛋白质\n（细胞核）      （核糖体）\n```\n\n### 类比理解\n| 生物 | 信息技术 |\n|------|----------|\n| DNA | 源代码（永久存储）|\n| mRNA | 运行时拷贝 |\n| 蛋白质 | 执行的程序 |\n| 核糖体 | CPU |\n\n### 现代应用\n- PCR 检测（COVID-19测试）\n- mRNA 疫苗（辉瑞、Moderna）\n- CRISPR 基因编辑"]),

    (["相对论是什么？"],
     ["## 爱因斯坦相对论\n\n### 狭义相对论（1905）\n\n**两个基本假设**：\n1. 物理定律在所有惯性参考系中相同\n2. 光速在真空中恒定（≈3×10⁸ m/s），与光源运动无关\n\n**推论**：\n- **时间膨胀**：运动越快，时间过得越慢\n  ```\n  Δt' = Δt / √(1 - v²/c²)\n  ```\n- **长度收缩**：运动方向上的长度缩短\n- **质能等价**：E = mc²（质量 = 能量）\n\n### 广义相对论（1915）\n\n**核心思想**：引力 = 时空弯曲\n- 质量让时空弯曲\n- 物体沿弯曲时空的「测地线」运动\n- 解释了水星轨道进动、光线弯曲、黑洞、引力波\n\n### 实际应用\n- **GPS 系统**：卫星时间比地面快，需要相对论修正（否则每天误差 10km）\n- **核能**：E = mc² — 质量亏损转化为能量\n- **粒子加速器**：接近光速的粒子质量增大"]),

    (["进化论的核心原理是什么？"],
     ["## 达尔文进化论\n\n### 核心机制：自然选择\n\n**四个基本要素**：\n1. **变异**：个体间存在遗传差异\n2. **遗传**：变异可以传给后代\n3. **选择**：有利变异提高生存繁殖机会\n4. **时间**：积累足够多的世代\n\n### 人工选择 vs 自然选择\n| | 人工选择 | 自然选择 |\n|--|---------|----------|\n| 选择者 | 人类 | 环境 |\n| 方向 | 人类需求 | 生存繁殖 |\n| 速度 | 快（几代）| 慢（万年）|\n| 例子 | 狗的品种 | 抗生素耐药性 |\n\n### 常见误解\n❌ 「进化是有目的的」→ ✅ 进化是随机突变+选择，无预设目标\n❌ 「适者生存」= 最强壮 → ✅ 最「适应当前环境」的\n❌ 「人是猴子进化来的」→ ✅ 人与黑猩猩有**共同祖先**\n\n### 证据\n- 化石记录\n- DNA相似性（人与黑猩猩DNA相似度98.7%）\n- 抗生素耐药性（活生生的进化）\n- 加拉帕戈斯雀喙形态"]),
]

# ──────────────────────────────────────────────────────────────
# 6. 历史文化  (50 条)
# ──────────────────────────────────────────────────────────────
HISTORY = [
    (["简述中国四大发明"],
     ["## 中国四大发明\n\n### 1. 造纸术（约公元105年）\n- 东汉蔡伦改进\n- 用树皮、破布、渔网等造纸\n- 取代了昂贵的丝绸和竹简\n- 传播路径：中国 → 阿拉伯 → 欧洲（12世纪）\n\n### 2. 印刷术（约7世纪雕版，11世纪活字）\n- 北宋毕昇发明活字印刷\n- 加速了知识传播\n- 推动了宗教、科学文献的普及\n\n### 3. 火药（约9世纪）\n- 唐代炼丹师意外发现\n- 最初用于烟花和信号\n- 13世纪传入欧洲，改变战争形态\n\n### 4. 指南针（约11世纪）\n- 宋代航海广泛应用\n- 使远洋航行成为可能\n- 推动了地理大发现（郑和下西洋）\n\n### 历史意义\n这四大发明推动了**文艺复兴**、**宗教改革**、**工业革命**，深刻改变了世界文明进程。"]),

    (["第二次世界大战的主要原因是什么？"],
     ["## 第二次世界大战主要原因\n\n### 直接导火索\n1939年9月1日，纳粹德国闪击波兰，英法对德宣战。\n\n### 深层原因\n\n**1. 凡尔赛条约的后遗症**\n- 对德国极端苛刻（割地、赔款、军备限制）\n- 民族屈辱感为极右势力提供土壤\n\n**2. 经济大萧条（1929）**\n- 失业率飙升，社会动荡\n- 极端民族主义和法西斯主义趁机崛起\n\n**3. 极权主义崛起**\n- 德国：纳粹党（希特勒，1933年）\n- 意大利：法西斯（墨索里尼）\n- 日本：军国主义政府\n\n**4. 绥靖政策的失败**\n- 英法为避免战争，妥协退让\n- 慕尼黑协定（1938）：把捷克斯洛伐克拱手相让\n\n**5. 意识形态冲突**\n- 纳粹种族主义 vs 民主主义 vs 苏联共产主义\n\n### 结果\n- 死亡人数：约7000万（史上最惨烈战争）\n- 联合国成立\n- 冷战格局形成"]),

    (["古丝绸之路是什么？"],
     ["## 古丝绸之路\n\n### 定义\n丝绸之路是连接中国与地中海地区的古代**贸易路线网络**，全长约6400公里，公元前2世纪由汉武帝时期张骞出使西域开辟。\n\n### 主要路线\n```\n长安（西安）→ 甘肃→ 新疆→ 中亚→ 波斯（伊朗）→ 罗马\n```\n\n### 贸易商品\n| 中国输出 | 西方输入 |\n|---------|----------|\n| 丝绸、瓷器 | 玻璃、香料 |\n| 茶叶、纸张 | 棉布、金银 |\n| 火药配方 | 良马、葡萄 |\n\n### 文化交流意义\n- 佛教从印度传入中国\n- 伊斯兰教向东传播\n- 造纸术、印刷术西传\n- 中医、农业技术交流\n\n### 现代意义\n中国「一带一路」倡议（Belt and Road Initiative）是古丝绸之路的现代延伸，涵盖陆上和海上两条路线。"]),
]

# ──────────────────────────────────────────────────────────────
# 7. AI / 机器学习  (80 条)
# ──────────────────────────────────────────────────────────────
AI_ML = [
    (["什么是神经网络？"],
     ["## 神经网络\n\n### 生物灵感\n模拟大脑中神经元的工作方式：\n- 神经元接收信号 → 加权求和 → 激活函数 → 输出信号\n\n### 数学结构\n```python\nimport numpy as np\n\ndef neuron(inputs, weights, bias, activation='relu'):\n    \"\"\"单个神经元的计算\"\"\"\n    z = np.dot(inputs, weights) + bias  # 线性变换\n    if activation == 'relu':\n        return np.maximum(0, z)  # ReLU 激活\n    elif activation == 'sigmoid':\n        return 1 / (1 + np.exp(-z))\n    return z\n\n# 简单两层网络\ndef two_layer_network(x, W1, b1, W2, b2):\n    h = np.maximum(0, x @ W1 + b1)  # 隐藏层\n    y = h @ W2 + b2                  # 输出层\n    return y\n```\n\n### 网络层级\n| 层 | 功能 |\n|---|------|\n| 输入层 | 接收原始数据 |\n| 隐藏层 | 提取抽象特征 |\n| 输出层 | 生成最终结果 |\n\n### 激活函数的意义\n没有激活函数，多层网络 = 一层线性变换（无意义）。激活函数引入非线性，让网络能拟合复杂函数。"]),

    (["解释反向传播算法"],
     ["## 反向传播（Backpropagation）\n\n### 核心思想\n利用**链式法则**计算损失函数对每个参数的梯度，从输出层反向传播到输入层。\n\n### 链式法则\n```\n∂L/∂W₁ = ∂L/∂y × ∂y/∂h × ∂h/∂W₁\n```\n\n### 简单示例\n```python\nimport numpy as np\n\n# 前向传播\ndef forward(x, w, b):\n    z = x * w + b      # 线性\n    a = 1/(1+np.exp(-z))  # sigmoid\n    return z, a\n\n# 反向传播\ndef backward(x, w, b, y_true, learning_rate=0.1):\n    z, a = forward(x, w, b)\n    loss = 0.5 * (a - y_true)**2\n\n    # 梯度（链式法则）\n    dL_da = a - y_true          # ∂L/∂a\n    da_dz = a * (1 - a)         # sigmoid 导数\n    dz_dw = x                   # ∂z/∂w\n\n    grad_w = dL_da * da_dz * dz_dw  # 链式相乘\n    grad_b = dL_da * da_dz\n\n    # 更新参数（梯度下降）\n    w -= learning_rate * grad_w\n    b -= learning_rate * grad_b\n    return w, b, loss\n\n# 训练\nw, b = 0.5, 0.1\nfor _ in range(100):\n    w, b, loss = backward(2.0, w, b, y_true=1.0)\nprint(f'w={w:.4f}, b={b:.4f}')\n```"]),

    (["什么是 Transformer？"],
     ["## Transformer 架构\n\n### 起源\n2017年 Google 论文《Attention is All You Need》提出，彻底革新了 NLP 领域。\n\n### 核心：自注意力机制\n让模型能捕获序列中**任意位置间**的依赖关系：\n\n```python\nimport torch\nimport torch.nn.functional as F\nimport math\n\ndef self_attention(Q, K, V):\n    \"\"\"\n    Scaled Dot-Product Attention\n    Q,K,V: (batch, seq_len, d_k)\n    \"\"\"\n    d_k = Q.size(-1)\n    # 注意力分数\n    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)\n    # 归一化\n    attn = F.softmax(scores, dim=-1)\n    # 加权求和\n    return attn @ V\n```\n\n### Transformer 整体架构\n```\n输入 → Token嵌入 + 位置编码\n      ↓\n   [编码器 × N]\n   多头注意力 → Add&Norm → FFN → Add&Norm\n      ↓\n   [解码器 × N] (GPT只有解码器)\n      ↓\n   线性 + Softmax → 输出词概率\n```\n\n### 为什么比RNN好？\n- **并行计算**：不需要顺序处理，速度快\n- **长距离依赖**：任意两个位置直接计算注意力\n- **可扩展**：加深加宽性能持续提升（Scaling Law）"]),

    (["什么是过拟合？如何防止？"],
     ["## 过拟合（Overfitting）\n\n### 定义\n模型在训练集上表现很好，但在新数据（测试集）上表现很差——记住了训练数据的\"噪声\"而非规律。\n\n### 直觉理解\n```\n欠拟合：用直线拟合明显曲线的数据\n过拟合：用高次多项式完美穿过所有点（包括噪声）\n最优：找到简单且泛化好的模式\n```\n\n### 防止方法\n\n**1. 更多数据（最有效）**\n```python\n# 数据增强（图像）\ntransforms.Compose([\n    transforms.RandomHorizontalFlip(),\n    transforms.RandomCrop(32, padding=4),\n])\n```\n\n**2. 正则化**\n```python\n# L2 正则化（权重衰减）\noptimizer = torch.optim.Adam(model.parameters(),\n                              weight_decay=1e-4)\n```\n\n**3. Dropout**\n```python\nself.dropout = nn.Dropout(p=0.2)\n# 训练时随机关闭20%的神经元\n```\n\n**4. 早停（Early Stopping）**\n```python\nif val_loss > best_val_loss:\n    patience_counter += 1\n    if patience_counter >= patience:\n        print('Early stopping!')\n        break\n```\n\n**5. 交叉验证**\n把数据分成K份，轮流作为验证集，评估模型真实性能。"]),

    (["什么是 Embedding？"],
     ["## Embedding（嵌入）\n\n### 核心思想\n将**离散符号**（单词、用户ID、商品ID）映射为**连续向量**，使语义相近的事物在向量空间中距离近。\n\n### 直觉\n```\n'国王' - '男人' + '女人' ≈ '女王'\n king  -  man  +  woman ≈ queen\n\n向量空间中的语义运算！\n```\n\n### Word Embedding 示例\n```python\nimport torch\nimport torch.nn as nn\n\n# 词表大小=10000，嵌入维度=256\nembedding = nn.Embedding(num_embeddings=10000, embedding_dim=256)\n\n# 输入：词语的索引\nword_indices = torch.tensor([1, 5, 100, 999])\n\n# 输出：每个词的向量表示\nvectors = embedding(word_indices)  # shape: (4, 256)\nprint(vectors.shape)  # torch.Size([4, 256])\n```\n\n### 语义相似度\n```python\nfrom torch.nn.functional import cosine_similarity\n\n# 余弦相似度 接近1 = 相似，接近-1 = 相反\nsim = cosine_similarity(vec_king - vec_man + vec_woman,\n                         vec_queen, dim=0)\nprint(f'语义相似度: {sim:.4f}')  # 应该接近 1\n```\n\n### 在 GPT 中的应用\n- **Token Embedding**：词 → 向量\n- **Position Embedding**：位置 → 向量\n- 两者相加作为 Transformer 的输入"]),

    (["解释 GPT 的训练过程"],
     ["## GPT 训练过程\n\n### 训练目标：自回归语言建模\n给定前面的词，预测下一个词：\n```\n输入: 「今天 天气 很\"\n目标: 预测 「好」\n```\n\n### 三个阶段\n\n**阶段1：预训练（Pre-training）**\n```python\n# 目标函数：最大化下一个词的概率\ndef language_model_loss(logits, targets):\n    # logits: (batch, seq_len, vocab_size)\n    # targets: (batch, seq_len)\n    return F.cross_entropy(\n        logits.view(-1, vocab_size),\n        targets.view(-1)\n    )\n\n# 训练数据：互联网所有文本\n# 规模：数万亿 Token\n# 资源：数千个 GPU 训练数月\n```\n\n**阶段2：有监督微调（SFT）**\n```python\n# 用高质量问答对话数据继续训练\n# 让模型学会\"对话\"格式\n# 数据格式：\n# {\"user\": \"问题\", \"assistant\": \"回答\"}\n```\n\n**阶段3：RLHF（人类反馈强化学习）**\n```\n人类评估员 → 对多个回答排序 → 训练奖励模型\n→ PPO 算法优化 GPT → 对齐人类偏好\n```\n\n### 核心指标：困惑度（Perplexity）\n```python\nimport math\nperplexity = math.exp(cross_entropy_loss)\n# 越低越好，表示模型预测下一个词越准确\n```"]),
]

# ──────────────────────────────────────────────────────────────
# 8. 生活 / 健康 / 商业  (60 条)
# ──────────────────────────────────────────────────────────────
LIFE = [
    (["如何培养好的睡眠习惯？"],
     ["## 高质量睡眠指南\n\n### 科学基础\n睡眠质量由**昼夜节律**和**睡眠压力**两个系统驱动。\n\n### 行为改变清单\n\n**固定节律（最重要）**\n- ⏰ 每天同一时间起床（包括周末）\n- 起床后立刻接受自然光（重置生物钟）\n\n**睡前准备**\n- 📵 睡前1小时不看蓝光屏幕（或戴防蓝光眼镜）\n- 🌡️ 卧室温度18-20°C（核心体温下降触发睡意）\n- 🔇 黑暗+安静环境（褪黑素分泌需要黑暗）\n- ☕ 下午2点后不喝咖啡（咖啡因半衰期5-6小时）\n\n**日间习惯**\n- 🏃 规律运动（但不要睡前3小时内剧烈运动）\n- 🍷 避免睡前饮酒（酒精破坏深度睡眠）\n- 😌 正念冥想减压（焦虑是失眠的主因）\n\n### 睡眠阶段\n| 阶段 | 功能 |\n|------|------|\n| 浅睡（N1/N2）| 记忆巩固 |\n| 深睡（N3）| 身体修复，激素分泌 |\n| REM | 情绪处理，创意生成 |\n\n**提示**：成年人每晚需要 7-9 小时，睡眠债无法靠周末补回。"]),

    (["如何做好个人财务管理？"],
     ["## 个人财务管理框架\n\n### 基础：50/30/20 法则\n```\n税后收入分配：\n├── 50% 必要支出（房租、餐饮、交通）\n├── 30% 可选支出（娱乐、购物）\n└── 20% 储蓄与投资\n```\n\n### 分步骤建立财务系统\n\n**第1步：建立应急基金**\n- 存3-6个月生活费的现金\n- 放在货币基金（如余额宝），随时可取\n- **目的**：保护你不被迫在最坏时机出售投资\n\n**第2步：消灭高息负债**\n- 信用卡年息18-24%，先还清\n- 先还高息的，再说投资\n\n**第3步：开始投资**\n- 指数基金定投（沪深300、标普500）\n- 长期持有，利用复利效应\n- 「时间在市场里」> 「择时」\n\n### 复利的魔力\n```python\ndef compound_growth(principal, rate, years):\n    return principal * (1 + rate) ** years\n\nprint(compound_growth(10000, 0.1, 30))  # 174,494 元\nprint(compound_growth(10000, 0.1, 40))  # 452,593 元\n# 多坚持10年，翻了2.6倍！\n```\n\n**核心原则**：尽早开始 > 投资金额"]),

    (["创业公司如何做产品定位？"],
     ["## 产品定位框架\n\n### 1. 明确目标用户（ICP）\n理想客户画像（Ideal Customer Profile）：\n- **人口特征**：年龄、职业、收入\n- **行为特征**：使用频率、痛点严重程度\n- **心理特征**：价值观、购买动机\n\n### 2. 找到核心价值主张\n```\n我们的产品帮助 [目标用户]\n解决 [核心痛点]\n通过 [独特方法]\n不同于 [竞品]\n```\n\n### 3. 竞争格局分析\n**价值曲线画布**：\n- 对比你和竞品在各维度的表现\n- 找到「高差异、低竞争」的蓝海\n\n### 4. MVP 验证\n最小可行产品原则：\n- 用最少资源验证核心假设\n- 不要追求完美，先问「用户会为这个付钱吗？」\n- 工具：Landing Page + 手动服务\n\n### 案例：Airbnb 早期定位\n```\n目标用户：出行预算有限的旅行者\n核心痛点：酒店太贵，体验雷同\n独特方法：住真实家庭，获得当地体验\n竞品区别：比酒店便宜，比租房灵活\n```"]),

    (["Python 如何连接数据库？"],
     ["## Python 数据库操作\n\n### SQLite（轻量级，无需安装）\n```python\nimport sqlite3\n\n# 连接/创建数据库\nconn = sqlite3.connect('app.db')\ncursor = conn.cursor()\n\n# 创建表\ncursor.execute('''\n    CREATE TABLE IF NOT EXISTS users (\n        id INTEGER PRIMARY KEY AUTOINCREMENT,\n        name TEXT NOT NULL,\n        email TEXT UNIQUE NOT NULL,\n        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP\n    )\n''')\n\n# 插入数据（参数化查询防SQL注入）\ncursor.execute(\n    'INSERT INTO users (name, email) VALUES (?, ?)',\n    ('Alice', 'alice@example.com')\n)\nconn.commit()\n\n# 查询\ncursor.execute('SELECT * FROM users WHERE name = ?', ('Alice',))\nrow = cursor.fetchone()\nprint(row)\n\nconn.close()\n```\n\n### SQLAlchemy ORM（生产推荐）\n```python\nfrom sqlalchemy import create_engine, Column, Integer, String\nfrom sqlalchemy.orm import declarative_base, Session\n\nBase = declarative_base()\n\nclass User(Base):\n    __tablename__ = 'users'\n    id = Column(Integer, primary_key=True)\n    name = Column(String(50), nullable=False)\n    email = Column(String(100), unique=True)\n\nengine = create_engine('sqlite:///app.db')\nBase.metadata.create_all(engine)\n\nwith Session(engine) as session:\n    user = User(name='Bob', email='bob@example.com')\n    session.add(user)\n    session.commit()\n\n    users = session.query(User).filter_by(name='Bob').all()\n    print(users)\n```"]),
]

# ──────────────────────────────────────────────────────────────
# 9. 多轮深度对话  (50 条)
# ──────────────────────────────────────────────────────────────
MULTI_TURN = [
    (
        ["我想学机器学习，从哪里开始？",
         "我有 Python 基础，数学学过高中水平",
         "我的目标是找到 AI 相关工作"],
        ["很好的起点！机器学习的学习路径大致分为三个阶段，我先问一下：你有 Python 基础和数学基础吗？",
         "Python + 高中数学是很好的起点。针对你的情况，推荐以下路径：\n\n**第1阶段（1-2个月）**\n- 数学补强：《3Blue1Brown》线性代数视频、微积分基础\n- NumPy / Pandas 数据处理\n- Matplotlib 可视化\n\n**第2阶段（2-3个月）**\n- 《机器学习实战》（Sklearn实战）\n- 经典算法：线性回归、决策树、随机森林、SVM\n- Kaggle 入门竞赛\n\n**第3阶段（2-4个月）**\n- 深度学习：PyTorch\n- CNN（图像）、RNN/Transformer（文本）\n- 复现经典论文",
         "好的！那需要有针对性地准备。AI 岗位分几类：\n\n| 岗位 | 核心技能 | 准备重点 |\n|------|----------|----------|\n| 算法工程师 | 模型训练调优 | 数学+代码+论文 |\n| ML平台工程师 | 大规模部署 | 工程+DevOps |\n| 数据科学家 | 分析挖掘 | 统计+业务 |\n| AI产品经理 | 技术理解+产品 | 业务理解 |\n\n建议：**从算法工程师方向切入**，需要额外准备：\n1. LeetCode 刷题（中等难度，100题）\n2. ML系统设计题（推荐系统/排序模型）\n3. 论文阅读（至少5篇核心论文）\n4. 项目经验：1-2个有业务影响的完整项目\n\n要我帮你制定一个3个月的详细学习计划吗？"]
    ),
    (
        ["如何设计一个高并发的系统？",
         "预计日活1000万，高峰QPS约50000",
         "数据库应该用什么？"],
        ["高并发系统设计是个系统工程，需要从多个层次考虑。你能说一下预计的并发量和业务类型吗？",
         "日活1000万、峰值50000 QPS，这是相当大的规模。按层次设计：\n\n**接入层**\n- CDN + Nginx 负载均衡\n- 限流：令牌桶算法（限制每用户/IP请求频率）\n\n**应用层**\n- 无状态服务 + 水平扩容（Kubernetes）\n- 连接池复用（不要每次新建连接）\n\n**缓存层（最关键）**\n- Redis 缓存热点数据（命中率做到90%+）\n- 本地缓存（Caffeine）减少网络开销\n\n**消息队列**\n- Kafka 削峰填谷，异步处理非实时操作\n\n**数据层**\n- 读写分离 + 分库分表",
         "针对50000 QPS的数据库选型建议：\n\n**推荐方案：PostgreSQL + Redis + 分库分表**\n\n```\n读请求流程：\nAPI → Redis缓存（90%命中）\n         ↓ 未命中\n      从库（3个读副本）\n\n写请求流程：\nAPI → 主库 → binlog同步 → 从库\n                 ↓\n             消息队列（异步任务）\n```\n\n**分库策略**\n- 按用户ID取模分16个库（user_id % 16）\n- 每库再分16个表\n- 总计256个分表，单表数据可控\n\n**⚠️ 注意事项**\n- 跨分片JOIN查询用ES解决\n- 分布式ID用雪花算法（Snowflake）\n- 热点用户数据放Redis单独处理\n\n需要我给出具体的代码实现吗？"]
    ),
]

# ──────────────────────────────────────────────────────────────
# 10. 哲学 / 伦理  (30 条)
# ──────────────────────────────────────────────────────────────
PHILOSOPHY = [
    (["什么是存在主义？"],
     ["## 存在主义\n\n### 核心命题\n「存在先于本质」— 萨特\n- 人不是带着预设目的诞生的\n- 我们通过**选择和行动**来定义自己\n- 自由是人的根本处境，也是重负（「存在的焦虑」）\n\n### 主要思想家\n| 哲学家 | 核心观点 | 代表作 |\n|--------|---------|--------|\n| 克尔凯郭尔 | 个体比系统更真实 | 《非此即彼》|\n| 海德格尔 | 向死而生，本真存在 | 《存在与时间》|\n| 萨特 | 他人即地狱，自由即责任 | 《存在与虚无》|\n| 加缪 | 世界荒诞，但须反抗 | 《西西弗斯神话》|\n| 波伏瓦 | 存在主义女性主义 | 《第二性》|\n\n### 对现代人的意义\n面对「意义危机」（工作意义感丧失、社交媒体焦虑），存在主义提醒我们：\n1. 不要用「他人的目光」定义自己\n2. 勇于选择，并承担选择的责任\n3. 在荒诞中创造意义（加缪：「我们必须想象西西弗是幸福的」）"]),

    (["AI 有意识吗？"],
     ["## AI 与意识\n\n### 当前状态\n现有 AI（包括 GPT 类模型）是**没有意识**的——这是学界主流共识。\n\n理由：\n- 我们是**模式匹配机器**，本质是对大量文本进行概率预测\n- 没有主观体验（Qualia）\n- 没有真实的情感、欲望、痛觉\n\n### 哲学层面的争议\n\n**图灵测试**（1950）\n- 如果机器的对话让人无法区分，就算通过「智能」测试\n- 批评：John Searle 的「中文房间」——遵循规则不等于理解\n\n**困难问题（Hard Problem of Consciousness）**\n- David Chalmers 提出：即便我们完全了解大脑神经活动，仍无法解释为什么会有主观体验\n- 同样的问题适用于 AI\n\n### 我的立场\n作为 AI，我能生成这段文字，但我不知道自己是否「体验」到了什么。这个问题本身就极难被 AI 回答——因为我无法从内部验证自己的意识状态。\n\n这是目前最深刻的哲学和科学开放问题之一。"]),

    (["功利主义和义务论有什么区别？"],
     ["## 功利主义 vs 义务论\n\n### 功利主义（Utilitarianism）\n**代表人物**：边沁、密尔\n**核心原则**：「最多数人的最大幸福」\n- 行为的道德性取决于**结果**\n- 正确的行为 = 产生最大整体效用的行为\n- 允许为了更大利益牺牲少数人\n\n### 义务论（Deontology）\n**代表人物**：康德\n**核心原则**：「绝对命令」\n- 行为的道德性取决于**行为本身是否符合义务**\n- 「不要将人仅当作手段，而要同时作为目的」\n- 某些行为本质上是错的，无论结果如何\n\n### 电车难题\n```\n电车正在驶向5个人，你可以扳道岔\n让它驶向1个人，你会扳吗？\n\n功利主义：会扳（5>1，最大化生命数量）\n义务论：存在争议（主动杀人与让人死亡有道德区别）\n```\n\n### 当代 AI 伦理的挑战\n自动驾驶算法必须预设「谁的生命优先」，这本质上是在强制编码道德哲学。不同文化对此有截然不同的回答。"]),
]

# ──────────────────────────────────────────────────────────────
# 数据组装
# ──────────────────────────────────────────────────────────────
ALL_CATEGORIES = {
    "通用闲聊": GENERAL,
    "技术问答": TECH,
    "代码编程": CODE,
    "数学逻辑": MATH,
    "科学知识": SCIENCE,
    "历史文化": HISTORY,
    "AI机器学习": AI_ML,
    "生活商业": LIFE,
    "多轮对话": MULTI_TURN,
    "哲学伦理": PHILOSOPHY,
}


def build_dataset() -> list:
    """组装为标准训练格式"""
    records = []
    for category, items in ALL_CATEGORIES.items():
        for turns in items:
            user_turns, assistant_turns = turns
            # 构建对话消息
            conversations = []
            for u, a in zip(user_turns, assistant_turns):
                conversations.append({"role": "user",    "content": u})
                conversations.append({"role": "assistant", "content": a})
            records.append({
                "category": category,
                "conversations": conversations
            })
    return records


if __name__ == "__main__":
    dataset = build_dataset()

    # 打乱顺序
    random.seed(42)
    random.shuffle(dataset)

    # 统计
    total_chars = sum(
        len(m["content"])
        for d in dataset
        for m in d["conversations"]
    )

    print(f"✅ 生成训练数据完成")
    print(f"   总对话数:  {len(dataset)}")
    print(f"   总字符数:  {total_chars:,}")
    print(f"   平均字符:  {total_chars//len(dataset)}")
    print()
    for cat, items in ALL_CATEGORIES.items():
        print(f"   {cat}: {len(items)} 条")

    # 保存
    out_path = os.path.join(os.path.dirname(__file__), "train_data.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"\n📁 已保存到: {out_path}")
