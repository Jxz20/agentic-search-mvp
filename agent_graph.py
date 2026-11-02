import os
import json
import time
import requests
from typing import TypedDict, List
from openai import OpenAI
from langgraph.graph import StateGraph, END

# ================= 1. 初始化与全局变量 =================
# ⚠️ 确保你的环境变量里有 ZHIPU_API_KEY，或者直接写死在这里
ZHIPU_KEY = os.getenv("ZHIPU_API_KEY")  
client = OpenAI(api_key=ZHIPU_KEY, base_url="https://open.bigmodel.cn/api/paas/v4/")

# 仪表盘：收集每一次调用的耗时和 Token
TRACE_LOG = []

# ================= 2. 定义 State =================
class AgentState(TypedDict):
    query: str             
    search_terms: List[str]
    papers: List[dict]     
    iteration: int         
    is_good: bool          
    feedback: str          # 错题本
    final_output: str      

# ================= 3. 定义 Node（带仪表盘和防弹衣） =================

# 工人1：规划者
def planner_node(state: AgentState):
    print(f"\n--- [Planner] 第 {state['iteration']+1} 次思考，正在拆解问题... ---")
    start_time = time.time()
    query = state["query"]
    feedback = state.get("feedback", "")
    
    history_context = ""
    if feedback:
        past_terms = ", ".join(state['search_terms'])
        history_context = f"\n⚠️历史失败教训：你上一次生成的搜索词是 [{past_terms}]，失败了。失败原因是：{feedback}。请避开这个思路，重新生成。\n"

    prompt = f"""用户的搜索请求是：'{query}'
{history_context}
请将其拆解为3个适合在英文学术数据库中搜索的关键词组合（用英文）。
要求：只输出一个合法的 JSON 数组（例如 ["term1", "term2", "term3"]），绝对不要输出任何解释性文字或 Markdown 标记。"""

    tokens_used = 0
    try:
        res = client.chat.completions.create(model="glm-4-flash", messages=[{"role": "user", "content": prompt}], temperature=0.3)
        content = res.choices[0].message.content.strip().replace("```json", "").replace("```", "")
        terms = json.loads(content)
        if not isinstance(terms, list): terms = [query]
        tokens_used = res.usage.total_tokens if hasattr(res, 'usage') else 0
    except Exception as e:
        print(f"Planner 解析失败，用原词兜底: {e}")
        terms = [query]
        
    latency = round(time.time() - start_time, 2)
    TRACE_LOG.append({"node": "planner", "iter": state["iteration"]+1, "latency(s)": latency, "tokens": tokens_used})
    
    print(f"-> 生成的搜索词: {terms} (耗时: {latency}s, Tokens: {tokens_used})")
    return {"search_terms": terms, "iteration": state["iteration"] + 1}

# 工人2：搜索者（穿上重试防弹衣）
def search_node(state: AgentState):
    print(f"--- [Search] 正在调用 OpenAlex API 检索... ---")
    start_time = time.time()
    all_papers = []
    url = "https://api.openalex.org/works"
    total_retries = 0
    
    for term in state["search_terms"]:
        params = {"search": term, "per-page": 5, "mailto": "xiuzhejia@zju.edu.cn"}
        
        # 防弹衣：最大重试 3 次
        max_retries = 3
        for attempt in range(max_retries):
            try:
                resp = requests.get(url, params=params, timeout=10)
                
                # 如果被限流，触发指数退避
                if resp.status_code == 429:
                    print(f"  ⚠️ OpenAlex 限流，等待 {2 ** attempt}s 后重试 ({attempt+1}/{max_retries})...")
                    time.sleep(2 ** attempt)
                    total_retries += 1
                    continue
                
                resp.raise_for_status() # 检查其他 4xx/5xx 错误
                
                # 解析数据
                data = resp.json()
                for item in data.get("results", []):
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
                break # 成功拿到数据，跳出重试循环，去搜下一个词
                
            except Exception as e:
                print(f"  ❌ 搜索 '{term}' 失败 ({attempt+1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    print(f"  🚨 彻底放弃 '{term}'，防止系统崩溃。")
                else:
                    time.sleep(1) # 网络错误短暂等待后重试
            
    unique_papers = {p["title"]: p for p in all_papers}.values()
    papers = list(unique_papers)[:10]
    
    latency = round(time.time() - start_time, 2)
    TRACE_LOG.append({"node": "search", "iter": state["iteration"], "latency(s)": latency, "retries": total_retries})
    
    print(f"-> 共搜到 {len(papers)} 篇独立论文 (耗时: {latency}s, 重试次数: {total_retries})")
    return {"papers": papers}

# 工人3：质检员
def evaluator_node(state: AgentState):
    print(f"--- [Evaluator] 正在评估结果质量... ---")
    start_time = time.time()
    
    if not state["papers"]:
        print("-> 一篇都没搜到，直接打回重做！")
        return {"is_good": False, "feedback": "上一轮检索没有返回任何论文。请检查搜索词是否拼写错误或过于复杂，请简化关键词。"}

    texts = "\n".join([f"[{i}] {p['title']} ({p.get('year')})" for i, p in enumerate(state["papers"])])
    prompt = f"""用户的原始意图是：'{state['query']}'
我们搜索到了以下论文：
{texts}

请判断：这些论文中，是否有至少 2 篇与用户的原始意图高度相关？
如果没有，或者全是跨学科的无关内容，请判定为不相关。
请严格只输出 JSON 格式： {{"is_good": true/false, "reason": "一句话理由"}}"""

    tokens_used = 0
    try:
        res = client.chat.completions.create(model="glm-4-flash", messages=[{"role": "user", "content": prompt}], temperature=0.1)
        content = res.choices[0].message.content.strip().replace("```json", "").replace("```", "")
        result = json.loads(content)
        is_good = result.get("is_good", False)
        reason = result.get("reason", "无理由")
        tokens_used = res.usage.total_tokens if hasattr(res, 'usage') else 0
    except Exception as e:
        is_good, reason = False, f"评估模型解析异常: {e}"
        
    latency = round(time.time() - start_time, 2)
    TRACE_LOG.append({"node": "evaluator", "iter": state["iteration"], "latency(s)": latency, "tokens": tokens_used})
    
    print(f"-> 评估结果: {is_good} | 理由: {reason} (耗时: {latency}s, Tokens: {tokens_used})")
    return {"is_good": is_good, "feedback": reason} 

# 工人4：包装员
def formatter_node(state: AgentState):
    print(f"--- [Formatter] 正在生成最终结果... ---")
    output = f"用户查询: {state['query']}\n\n"
    output += "为您找到以下相关论文：\n"
    for i, p in enumerate(state["papers"], 1):
        output += f"{i}. {p['title']} ({p.get('year')})\n"
    return {"final_output": output}

# ================= 4. 拼装流水线（Graph） =================
workflow = StateGraph(AgentState)
workflow.add_node("planner", planner_node)
workflow.add_node("search", search_node)
workflow.add_node("evaluator", evaluator_node)
workflow.add_node("formatter", formatter_node)

workflow.set_entry_point("planner")
workflow.add_edge("planner", "search")
workflow.add_edge("search", "evaluator")
workflow.add_edge("formatter", END)

def route_after_eval(state: AgentState):
    if state["is_good"] or state["iteration"] >= 3:
        return "formatter"
    return "planner"

workflow.add_conditional_edges("evaluator", route_after_eval)
app = workflow.compile()

# ================= 5. 运行入口 =================
if __name__ == "__main__":
    initial_state = {
        "query": "用强化学习解决大模型幻觉的论文", # 你可以换成一些偏门的词测试重试和打回
        "search_terms": [],
        "papers": [],
        "iteration": 0,
        "is_good": False,
        "feedback": "",
        "final_output": ""
    }
    
    print("🚀 启动 Agentic Search...")
    result = app.invoke(initial_state)
    
    print("\n========== 最终输出 ==========")
    print(result["final_output"])
    
    # 打印仪表盘数据（这就是你简历里“成本与延迟监控”的铁证）
    print("\n========== 运行仪表盘 (Trace Log) ==========")
    total_tokens = 0
    total_time = 0
    for log in TRACE_LOG:
        print(log)
        total_time += log.get("latency(s)", 0)
        total_tokens += log.get("tokens", 0)
        
    print("-" * 40)
    print(f"📊 汇总: 总耗时 {round(total_time, 2)} 秒 | 总 Token 消耗 {total_tokens} | 循环总次数 {result['iteration']}")