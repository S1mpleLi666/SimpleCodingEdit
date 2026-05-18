import re, sys, os
from io import StringIO
from typing import TypedDict, List, Annotated, Dict, Any

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# -------- 工具 --------
class PythonExecutor:
    @staticmethod
    def run(code: str) -> Dict[str, Any]:
        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()
        local_scope = {}
        try:
            exec(code, {"__builtins__": __builtins__}, local_scope)
            output = mystdout.getvalue()
            # 提取所有变量（排除内置名称和可调用对象）
            variables = {}
            for k, v in local_scope.items():
                if not k.startswith('_') and not callable(v):
                    try:
                        # 尝试序列化，处理可打印的类型
                        repr(v)
                        variables[k] = v
                    except:
                        pass
            return {"output": output, "error": "", "variables": variables}
        except Exception as e:
            return {"output": "", "error": str(e), "variables": {}}
        finally:
            sys.stdout = old_stdout

# -------- 状态 --------
class AgentState(TypedDict, total=False):
    messages: Annotated[List, add_messages]
    plan: str
    code: str
    output: str
    error: str
    variables: Dict[str, Any]
    feedback: str
    needs_fix: bool
    iterations: int

# -------- LLM --------
llm = ChatOpenAI(model="Qwen3.5-122B-A10B",
                temperature=0.6,
                base_url=os.environ["BASE_URL"],
                api_key=os.environ["OPENAI_API_KEY"])

# -------- 节点 --------
def planner(state: AgentState) -> AgentState:
    prompt = """你是一个编程助手，需要解决以下问题。
请用中文简要说明你将如何分步骤完成这个任务（仅输出计划，不要写代码）。

注意：如果用户只是打招呼（如"你好"、"hi"等），不需要写代码，只需说明会礼貌回应即可。

问题：{input}
计划："""
    user_msg = state["messages"][-1].content
    formatted = prompt.format(input=user_msg)
    response = llm.invoke(formatted)
    return {"plan": response.content, "iterations": 0}

def coder(state: AgentState) -> AgentState:
    plan = state.get("plan", "")
    feedback = state.get("feedback", "")
    error = state.get("error", "")

    # 构建提示信息
    hints = []
    if error:
        hints.append(f"上次代码执行出错：{error}")
    if feedback and "执行成功" not in feedback:
        hints.append(feedback)

    error_hint = "\n".join(hints) if hints else ""
    if error_hint:
        error_hint = f"\n请根据以下问题修正代码：\n{error_hint}"

    prompt = f"""你是一个 Python 专家。根据以下计划写出可执行代码。
{plan}
{error_hint}
要求：代码直接可运行，用 print() 输出结果。仅输出代码，不要解释。"""
    response = llm.invoke(prompt)
    code = re.sub(r"```python\s*|```", "", response.content).strip()
    return {"code": code}

def executor(state: AgentState) -> AgentState:
    result = PythonExecutor.run(state["code"])
    return {
        "output": result["output"],
        "error": result["error"],
        "variables": result["variables"],
        "iterations": state["iterations"] + 1
    }

def reflector(state: AgentState) -> AgentState:
    """分析执行结果，判断是否需要修正"""
    error = state.get("error", "")
    output = state.get("output", "")

    feedback = ""
    needs_fix = False

    if error:
        needs_fix = True
        feedback = f"代码出错：{error}"
    elif not output.strip():
        # 检查代码是否包含 input() - 这是不合适的，因为程序已经在等待输入
        code = state.get("code", "")
        if "input(" in code:
            needs_fix = True
            feedback = "代码使用了 input() 函数，这会导致程序阻塞。请直接使用 print() 输出结果，不要等待用户输入。"
        else:
            needs_fix = True
            feedback = "代码未产生任何输出，请检查是否有 print() 语句"
    else:
        feedback = "执行成功"

    return {"needs_fix": needs_fix, "feedback": feedback}

def should_continue(state: AgentState) -> str:
    MAX_ITER = 3
    if state.get("needs_fix") and state["iterations"] < MAX_ITER:
        return "coder"
    return "__end__"

# -------- 构建图 --------
workflow = StateGraph(AgentState)
workflow.add_node("planner", planner)
workflow.add_node("coder", coder)
workflow.add_node("executor", executor)
workflow.add_node("reflector", reflector)

workflow.set_entry_point("planner")
workflow.add_edge("planner", "coder")
workflow.add_edge("coder", "executor")
workflow.add_edge("executor", "reflector")
workflow.add_conditional_edges(
    "reflector", should_continue,
    {"coder": "coder", "__end__": END}
)
app = workflow.compile()

# -------- 运行 --------
if __name__ == "__main__":
    if len(sys.argv) > 1:
        user_request = " ".join(sys.argv[1:])
    else:
        user_request = input("请输入需求：")

    inputs = {"messages": [HumanMessage(content=user_request)]}
    for output in app.stream(inputs):
        for key, value in output.items():
            if key == "reflector":
                continue
            print(f"\n--- {key} ---")
            for k, v in value.items():
                if v:
                    print(f"{k}: {v}")
