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
        try:
            exec(code, {"__builtins__": __builtins__})
            result = mystdout.getvalue()
            return {"output": result, "error": ""}
        except Exception as e:
            return {"output": "", "error": str(e)}
        finally:
            sys.stdout = old_stdout
 
# -------- 状态 --------
class AgentState(TypedDict):
    messages: Annotated[List, add_messages]
    plan: str
    code: str
    output: str
    error: str
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
问题：{input}
计划："""
    user_msg = state["messages"][-1].content
    formatted = prompt.format(input=user_msg)
    response = llm.invoke(formatted)
    return {"plan": response.content, "iterations": 0}
 
def coder(state: AgentState) -> AgentState:
    plan = state.get("plan", "")
    error_hint = ""
    if state.get("error"):
        error_hint = f"\n上次代码执行出错：{state['error']}\n请修正代码。"
    prompt = f"""你是一个 Python 专家。根据以下计划和错误信息，写出可执行代码。
{plan}
{error_hint}
要求：代码要直接可运行，用 print() 输出结果。仅输出代码，不要解释。"""
    response = llm.invoke(prompt)
    code = response.content 
    code = re.sub(r"```python\s*|```", "", code).strip()
    return {"code": code}
 
def executor(state: AgentState) -> AgentState:
    result = PythonExecutor.run(state["code"])
    return {
        "output": result["output"],
        "error": result["error"],
        "iterations": state["iterations"] + 1
    }
 
def reflector(state: AgentState) -> AgentState:
    return {}
 
def should_continue(state: AgentState) -> str:
    MAX_ITER = 3
    if state.get("error") and state["iterations"] < MAX_ITER:
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
    user_request = "帮我写一个斐波那契数列函数，计算前10项，并打印出来。"
    inputs = {"messages": [HumanMessage(content=user_request)]}
    for output in app.stream(inputs):
        for key, value in output.items():
            if key == "reflector":
                continue
            print(f"\n--- {key} ---")
            for k, v in value.items():
                if v:
                    print(f"{k}: {v}")