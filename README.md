🤖 基于 LangGraph 的编程 Agent

一个使用 LangChain + LangGraph 从零构建的代码编程 Agent，能够根据自然语言需求自动生成 Python 代码、执行并自我修复错误。

🎯 功能特性

- 🧠 自然语言理解：用中文描述需求，Agent 自动分析并规划解决方案
- ✍️ 自动代码生成：基于 LLM 生成可执行的 Python 代码
- ⚡ 代码执行：自动运行代码并捕获输出、错误和**变量值**
- 🔄 自我修复循环：代码出错或无输出时自动分析并重试修正
- 🔍 智能反思：reflector 节点分析执行结果，判断是否需要修正
- 📊 流式状态追踪：可观察每个节点的执行过程和中间结果
- 🛑 智能终止条件：最大迭代次数限制，避免无限循环浪费 token

🏗️ 架构设计

Agent 采用 Plan-and-Execute + Self-Debugging 模式，通过 LangGraph 的 StateGraph 实现多节点循环流程：

```
         ┌─────────┐
         │  start  │
         └────┬────┘
              ▼ 
       ┌──────────────┐
       │  planner     │  ← 分析需求，输出执行计划 
       └──────┬───────┘
              ▼ 
       ┌──────────────┐
       │  coder       │  ← 根据计划生成 Python 代码
       └──────┬───────┘
              ▼
       ┌──────────────┐ 
       │  executor    │  ← 执行代码，捕获输出/错误/变量
       └──────┬───────┘
              ▼
       ┌──────────────┐
       │  reflector   │  ← 判断结果是否正确，生成反馈
       └──┬────────┬──┘ 
     ok? │        │ fail → 返回 coder
         ▼        │
     ┌──────┐     │
     │ end  │◄────┘
     └──────┘
```

节点说明

| 节点 | 职责 |
|------|------|
| Planner | 解析用户需求，生成分步骤的自然语言计划 |
| Coder | 根据计划和错误/反馈信息生成/修复 Python 代码 |
| Executor | 执行代码，捕获 stdout、异常和**所有变量值** |
| Reflector | 评估执行结果，生成反馈并决定是否继续修复 |

状态设计

所有节点共享同一份 AgentState，实现状态累积传递：

```python
class AgentState(TypedDict, total=False):
    messages:    Annotated[List, add_messages]  # 对话历史
    plan:        str                            # 执行计划
    code:        str                            # 当前代码
    output:      str                            # 执行输出 
    error:       str                            # 错误信息
    variables:   Dict[str, Any]                 # 提取的变量值 ✨
    feedback:    str                            # reflector 反馈 ✨
    needs_fix:   bool                           # 是否需要修正 ✨
    iterations:  int                            # 当前尝试次数
```

📦 安装

环境要求

- Python >= 3.10
- 支持 OpenAI 兼容 API 的 LLM 服务（OpenAI / DeepSeek / 通义千问 / 本地部署等）

安装依赖

```bash
pip install langchain-openai langgraph python-dotenv
```

🚀 快速开始

1. 配置环境变量

在项目根目录创建 `.env` 文件：

```env
OPENAI_API_KEY=你的 API 密钥
BASE_URL=http://你的 API 地址/v1   # 如果不是使用 OpenAI 官方，需指定
```

示例 — 使用 Qwen 或其他兼容 API：

```env
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx
BASE_URL=https://api.example.com/v1
```

2. 运行 Agent

```bash
python main.py
```

或者使用命令行参数直接传入需求：

```bash
python main.py "写一个函数计算 1 到 100 的总和"
```

或者创建你自己的脚本：

```python
from langchain_core.messages import HumanMessage
from main import app

user_request = "帮我写一个斐波那契数列函数，计算前 10 项，并打印出来。"
inputs = {"messages": [HumanMessage(content=user_request)]}

for output in app.stream(inputs):
    for key, value in output.items():
        if key == "reflector":
            continue 
        print(f"\n--- {key} ---")
        for k, v in value.items():
            if v:
                print(f"{k}: {v}")
```

3. 观察输出

```
--- planner ---
plan: 1. 定义斐波那契函数  2. 计算前 10 项  3. 打印结果

--- coder ---
code: def fib(n):
    a, b = 0, 1
    for _ in range(n):
        print(a, end=' ')
        a, b = b, a + b
fib(10)

--- executor ---
output: 0 1 1 2 3 5 8 13 21 34
variables: {'a': 55, 'b': 89}
iterations: 1
```

⚙️ 配置参数

可以在代码中调整以下参数：

| 参数 | 位置 | 默认值 | 说明 |
|------|------|--------|------|
| model | ChatOpenAI() | Qwen3.5-122B-A10B | LLM 模型名称 |
| temperature | ChatOpenAI() | 0.6 | 生成温度（越低越稳定） |
| MAX_ITER | should_continue() | 3 | 最大重试次数 |
| base_url | ChatOpenAI() | 环境变量 | API 地址 |

🔧 扩展方向

1. 添加新工具

```python
from langchain.tools import Tool

tools = [
    Tool(name="python_executor", func=PythonExecutor.run, description="执行 Python 代码"),
    Tool(name="websearch", func=search_function, description="搜索网络获取信息"),
    Tool(name="filewriter", func=write_file, description="将代码写入文件"),
]
```

2. 更安全的代码执行

```python
import subprocess, tempfile 

def safe_run(code: str) -> dict:
    # 写入临时文件，用 subprocess 隔离执行
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
        f.write(code.encode())
    try:
        result = subprocess.run(["python", f.name], capture_output=True, timeout=10)
        return {"output": result.stdout.decode(), "error": result.stderr.decode()}
    except subprocess.TimeoutExpired:
        return {"output": "", "error": "执行超时"}
```

3. 引入人类审核（Human-in-the-Loop）

```python
# 在 coder 和 executor 之间加入人工确认节点
workflow.add_node("human_review", human_review_node)
workflow.add_edge("coder", "human_review")
# human_review_node 中使用 interrupt 等待人工确认
```

4. 升级为真正的 ReAct 模式

```python
from langchain.agents import create_react_agent, AgentExecutor 

agent = create_react_agent(llm, tools, react_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)
```

📁 项目结构

```
.
├── main.py          # 主程序（Agent 图定义 + 运行入口）
├── .env             # 环境变量（API Key & Base URL）
├── README.md        # 项目文档
└── requirements.txt # 依赖清单
```

🧪 示例请求

| 输入 | 预期行为 |
|------|----------|
| "写一个冒泡排序并测试" | 生成排序代码，执行并打印结果 |
| "计算 1 到 100 的质数" | 生成质数筛法，打印列表 |
| "用 matplotlib 画一个 sin 曲线" | 生成绘图代码，保存图片（需扩展文件工具） |
| "故意写一个语法错误的程序" | 多次尝试修复，最终报错或成功 |

❓ 常见问题

**Q: 报错 ModuleNotFoundError: No module named 'langchain.schema'**

A: LangChain 新版已将消息类迁移到 `langchain_core`，将 `from langchain.schema import HumanMessage` 改为 `from langchain_core.messages import HumanMessage`。

**Q: 报错 ConnectTimeout 或 APITimeoutError**

A: 网络无法连接 API 服务器。检查 `.env` 中的 `BASE_URL` 是否正确，或配置代理。

**Q: 如何更换其他 LLM？**

A: 修改 `ChatOpenAI` 的 `model` 和 `base_url` 参数即可，兼容所有 OpenAI 格式的 API。

**Q: 代码执行不安全怎么办？**

A: 生产环境建议使用 Docker 沙箱、subprocess 隔离或云函数执行，避免直接在主进程 `exec()`。

📄 许可

MIT License

🙏 参考

- [LangGraph 官方文档](https://langchain-ai.github.io/langgraph/)
- [LangChain 官方文档](https://python.langchain.com/)
- [ReAct 论文](https://arxiv.org/abs/2210.03629)
