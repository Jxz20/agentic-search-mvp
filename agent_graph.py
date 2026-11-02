import os
import json
import requests
from typing import TypedDict, List
from openai import OpenAI
from langgraph.graph import StateGraph, END

# ================= 1. 初始化 =================
# ⚠️ 把这里换成你的真实智谱 Key！
ZHIPU_KEY = os.getenv("ZHIPU_API_KEY")  
client = OpenAI(api_key=ZHIPU_KEY, base_url="https://open.bigmodel.cn/api/paas/v4/")

# ================= 2. 定义 State（传送带上的大箱子） =================
class AgentState(TypedDict):
    query: str             # 用户的原始提问
    search_terms: List[str]# 大模型想出来的搜索词
    papers: List[dict]     # 搜回来的论文
    iteration: int         # 当前尝试了几次
    is_good: bool          # 结果好不好？
    feedback: str  # 新增：存储上一次失败的原因
    final_output: str      # 最终汇报文本

# ================= 3. 定义 Node（干活的工人） =================

# 工人1：规划者
def planner_node(state: AgentState):
    print(f"\n--- [Planner] 第 {state['iteration']+1} 次思考，正在拆解问题... ---")
    query = state["query"]
    feedback = state.get("feedback", "")
    
    history_context = ""
    if feedback:
        # 把 list 转成字符串，避免模型解析混乱
        past_terms = ", ".join(state['search_terms'])
        history_context = f"\n⚠️历史失败教训：你上一次生成的搜索词是 [{past_terms}]，失败了。失败原因是：{feedback}。请避开这个思路，重新生成。\n"

    prompt = f"""用户的搜索请求是：'{query}'
{history_context}
请将其拆解为3个适合在英文学术数据库中搜索的关键词组合（用英文）。
要求：只输出一个合法的 JSON 数组（例如 ["term1", "term2", "term3"]），绝对不要输出任何解释性文字或 Markdown 标记。"""

    try:
        res = client.chat.completions.create(model="glm-4-flash", messages=[{"role": "user", "content": prompt}], temperature=0.3) # 稍微调高一点点温度，让它改变思路
        content = res.choices[0].message.content.strip().replace("```json", "").replace("```", "")
        terms = json.loads(content)
        if not isinstance(terms, list): terms = [query]
    except Exception as e:
        print(f"Planner 解析失败，用原词兜底: {e}")
        terms = [query]
        
    print(f"-> 生成的搜索词: {terms}")
    return {"search_terms": terms, "iteration": state["iteration"] + 1}

# 工人2：搜索者（去 OpenAlex 抓论文）
def search_node(state: AgentState):
    print(f"--- [Search] 正在调用 OpenAlex API 检索... ---")
    all_papers = []
    url = "https://api.openalex.org/works"
    
    for term in state["search_terms"]:
        params = {"search": term, "per-page": 5, "mailto": "xiuzhejia@zju.edu.cn"}
        try:
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                for item in data.get("results", []):
                    # 简单处理摘要
                    abstract_inv = item.get("abstract_inverted_index")
                    abstract = ""
                    if abstract_inv:
                        words = sorted([(pos, w) for w, positions in abstract_inv.items() for pos in positions])
                        abstract = " ".join([w[1] for w in words])[:500]
                    
                    all_papers.append({
                        "title": item.get("title") or "无标题",
                        "abstract": abstract,
                        "year": item.get("publication_year"),
                    })
        except Exception as e:
            print(f"搜索 '{term}' 失败: {e}")
            
    # 简单按标题去重，保留前 10 篇
    unique_papers = {p["title"]: p for p in all_papers}.values()
    papers = list(unique_papers)[:10]
    print(f"-> 共搜到 {len(papers)} 篇独立论文")
    return {"papers": papers}

# 工人3：质检员（评估搜回来的东西对不对）
def evaluator_node(state: AgentState):
    print(f"--- [Evaluator] 正在评估结果质量... ---")
    if not state["papers"]:
        print("-> 一篇都没搜到，直接打回重做！")
        # 修复 Bug 1：这里必须带上反馈
        return {"is_good": False, "feedback": "上一轮检索没有返回任何论文。请检查搜索词是否拼写错误或过于复杂，请简化关键词。"}

    texts = "\n".join([f"[{i}] {p['title']} ({p.get('year')})" for i, p in enumerate(state["papers"])])
    prompt = f"""用户的原始意图是：'{state['query']}'
我们搜索到了以下论文：
{texts}

请判断：这些论文中，是否有至少 2 篇与用户的原始意图高度相关？
如果没有，或者全是跨学科的无关内容，请判定为不相关。
请严格只输出 JSON 格式： {{"is_good": true/false, "reason": "一句话理由"}}"""

    try:
        res = client.chat.completions.create(model="glm-4-flash", messages=[{"role": "user", "content": prompt}], temperature=0.1)
        content = res.choices[0].message.content.strip().replace("```json", "").replace("```", "")
        result = json.loads(content)
        is_good = result.get("is_good", False)
        reason = result.get("reason", "无理由")
        print(f"-> 评估结果: {is_good} | 理由: {reason}")
        
        # 这里把你的 feedback 机制完美保留
        return {"is_good": is_good, "feedback": reason} 
    except Exception as e:
        print(f"评估出错，默认打回: {e}")
        return {"is_good": False, "feedback": "评估模型解析异常"}

# 工人4：包装员（整理最终输出）
def formatter_node(state: AgentState):
    print(f"--- [Formatter] 正在生成最终结果... ---")
    output = f"用户查询: {state['query']}\n\n"
    output += "为您找到以下相关论文：\n"
    for i, p in enumerate(state["papers"], 1):
        output += f"{i}. {p['title']} ({p.get('year')})\n"
    return {"final_output": output}

# ================= 4. 拼装流水线（Graph） =================
workflow = StateGraph(AgentState)

# 注册工人
workflow.add_node("planner", planner_node)
workflow.add_node("search", search_node)
workflow.add_node("evaluator", evaluator_node)
workflow.add_node("formatter", formatter_node)

# 设定从哪开始
workflow.set_entry_point("planner")

# 设定轨道（普通连线）
workflow.add_edge("planner", "search")
workflow.add_edge("search", "evaluator")
workflow.add_edge("formatter", END)

# 设定质检员之后的路线（条件连线：走循环还是直接下班？）
def route_after_eval(state: AgentState):
    # 如果质量好，或者已经试了3次了，就收工
    if state["is_good"] or state["iteration"] >= 3:
        return "formatter"
    # 否则，打回给规划员重新想词
    return "planner"

workflow.add_conditional_edges("evaluator", route_after_eval)

# 编译成可运行的机器
app = workflow.compile()

# ================= 5. 运行入口 =================
if __name__ == "__main__":
    # 初始化状态箱子
    initial_state = {
        "query": "用强化学习解决大模型幻觉的论文",
        "search_terms": [],
        "papers": [],
        "iteration": 0,
        "is_good": False,
        "final_output": ""
    }
    # 启动流水线
    result = app.invoke(initial_state)
    print("\n========== 最终输出 ==========")
    print(result["final_output"])