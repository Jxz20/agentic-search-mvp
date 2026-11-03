import os
import json
import time
import requests
import urllib.parse
import xml.etree.ElementTree as ET
from typing import TypedDict, List
from openai import OpenAI
from langgraph.graph import StateGraph, END

# ================= 1. 模型抽象层 =================
class LLMClient:
    def __init__(self):
        self.provider = os.getenv("LLM_PROVIDER", "zhipu").lower()
        if self.provider == "deepseek":
            self.api_key = os.getenv("DEEPSEEK_API_KEY")
            self.base_url = "https://api.deepseek.com/v1"
            self.model = "deepseek-chat"
        else:
            self.api_key = os.getenv("ZHIPU_API_KEY")
            self.base_url = "https://open.bigmodel.cn/api/paas/v4/"
            self.model = "glm-4-flash"
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def chat(self, prompt: str, temperature: float = 0.1) -> tuple:
        try:
            res = self.client.chat.completions.create(
                model=self.model, messages=[{"role": "user", "content": prompt}], temperature=temperature
            )
            return res.choices[0].message.content.strip(), res.usage.total_tokens if hasattr(res, 'usage') else 0
        except Exception as e:
            print(f"[LLM Error] {e}")
            return "", 0

llm = LLMClient()
TRACE_LOG = []

# ================= 2. 断路器与多源级联 =================
class ProviderState:
    def __init__(self, name, max_failures=3, cooldown=60):
        self.name, self.failures, self.max_failures, self.cooldown = name, 0, max_failures, cooldown
        self.last_failure_time, self.open = 0, False
    def record_success(self): self.failures, self.open = 0, False
    def record_failure(self):
        self.failures += 1
        self.last_failure_time = time.time()
        if self.failures >= self.max_failures:
            self.open = True
            print(f"    [CircuitBreaker] ⚠️ {self.name} 已熔断，{self.cooldown}s 内跳过。")
    def can_attempt(self):
        if not self.open: return True
        if time.time() - self.last_failure_time > self.cooldown:
            self.open, self.failures = False, 0
            return True
        return False

CROSSREF_STATE = ProviderState("Crossref", max_failures=3, cooldown=60)
ARXIV_STATE = ProviderState("arXiv", max_failures=3, cooldown=120)

HEADERS = {"User-Agent": "AcademicAgent/1.0 (mailto:xiuzhejia@zju.edu.cn)"}

# ================= 3. 多源检索与引文扩展 =================
def fetch_crossref(term, limit=5, expand_citations=False):
    if not CROSSREF_STATE.can_attempt(): return None
    try:
        params = {"query.bibliographic": term, "rows": limit, "select": "title,abstract,issued,DOI,reference"}
        resp = requests.get("https://api.crossref.org/works", params=params, headers=HEADERS, timeout=8)
        if resp.status_code != 200:
            CROSSREF_STATE.record_failure()
            return None
        
        items = resp.json().get("message", {}).get("items", [])
        papers = []
        for item in items:
            title = (item.get("title") or ["无标题"])[0]
            abstract = (item.get("abstract", "") or "").replace("<jats:p>", "").replace("</jats:p>", "")[:500]
            year = item["issued"]["date-parts"][0][0] if item.get("issued", {}).get("date-parts") else None
            papers.append({"title": title, "abstract": abstract, "year": year, "source": "crossref", "doi": item.get("DOI")})
            
            # 🌟 引文扩展逻辑（如果开启）
            if expand_citations and item.get("reference"):
                refs = item["reference"][:2] # 最多取 2 篇参考文献防止太长
                for ref in refs:
                    ref_title = ref.get("article-title") or ref.get("unstructured")
                    if ref_title:
                        papers.append({
                            "title": ref_title, "abstract": "", "year": None, "source": "citation_expand"
                        })
        CROSSREF_STATE.record_success()
        return papers
    except Exception as e:
        CROSSREF_STATE.record_failure()
        return None

def fetch_arxiv(term, limit=5):
    if not ARXIV_STATE.can_attempt(): return None
    try:
        query = f'search_query=all:"{term}"&start=0&max_results={limit}'
        resp = requests.get("http://export.arxiv.org/api/query?" + urllib.parse.quote(query), headers=HEADERS, timeout=10)
        if resp.status_code != 200:
            ARXIV_STATE.record_failure()
            return None
        root = ET.fromstring(resp.content)
        papers = []
        for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
            papers.append({
                "title": entry.find('{http://www.w3.org/2005/Atom}title').text.strip().replace('\n', ' '),
                "abstract": entry.find('{http://www.w3.org/2005/Atom}summary').text.strip().replace('\n', ' ')[:500],
                "year": entry.find('{http://www.w3.org/2005/Atom}published').text[:4],
                "source": "arxiv"
            })
        ARXIV_STATE.record_success()
        return papers
    except Exception:
        ARXIV_STATE.record_failure()
        return None

def cascade_search(term, limit=5, expand_citations=False):
    # 优先尝试 Crossref（带引文扩展）
    papers = fetch_crossref(term, limit, expand_citations=expand_citations)
    if papers: return papers
    # 降级到 arXiv
    papers = fetch_arxiv(term, limit)
    if papers: return papers
    return []

# ================= 4. Node 定义 =================
class AgentState(TypedDict):
    query: str; search_terms: List[str]; papers: List[dict]; iteration: int; is_good: bool; feedback: str; final_output: str

def planner_node(state: AgentState):
    print(f"\n--- [Planner] 第 {state['iteration']+1} 次思考 ---")
    feedback = state.get("feedback", "")
    history = f"\n⚠️历史教训：上次搜索词 {', '.join(state['search_terms'])} 失败。原因：{feedback}。\n" if feedback else ""
    prompt = f"搜索请求：'{state['query']}'\n{history}\n拆解为3个英文关键词。只输出 JSON 数组。"
    
    content, tokens = llm.chat(prompt, temperature=0.3)
    try:
        terms = json.loads(content.replace("```json", "").replace("```", ""))
        if not isinstance(terms, list): terms = [state['query']]
    except Exception:
        terms = [state['query']]
    print(f"-> 搜索词: {terms} (Tokens: {tokens})")
    TRACE_LOG.append({"node": "planner", "tokens": tokens})
    
    return {"search_terms": terms, "iteration": state["iteration"] + 1}

def search_node(state: AgentState):
    print(f"--- [Search] 多源级联检索 (包含引文扩展) ---")
    start_time = time.time()
    all_papers = []
    
    # 判断是否开启引文扩展（只有第一轮开启，防止后续无限膨胀）
    expand = (state["iteration"] == 0)
    
    for term in state["search_terms"]:
        papers = cascade_search(term, limit=5, expand_citations=expand)
        all_papers.extend(papers)
        time.sleep(1)
        
    unique_papers = {p["title"].lower(): p for p in all_papers if p["title"]}.values()
    papers = list(unique_papers)[:15]
    
    # 统计各来源数量
    sources = {}
    for p in papers:
        sources[p["source"]] = sources.get(p["source"], 0) + 1
        
    latency = round(time.time() - start_time, 2)
    TRACE_LOG.append({"node": "search", "latency(s)": latency})
    print(f"-> 候选池: {len(papers)} 篇 | 来源分布: {sources} (耗时: {latency}s)")
    return {"papers": papers}

def evaluator_node(state: AgentState):
    print(f"--- [Evaluator] 评估结果 ---")
    if not state["papers"]:
        return {"is_good": False, "feedback": "全网无结果。"}
    
    texts = "\n".join([f"[{i}] {p['title']} ({p.get('year')})" for i, p in enumerate(state["papers"])])
    prompt = f"意图：'{state['query']}'\n论文：\n{texts}\n是否有至少2篇高度相关？严格输出JSON： {{\"is_good\": true/false, \"reason\": \"理由\"}}"
    
    content, tokens = llm.chat(prompt, temperature=0.1)
    try:
        result = json.loads(content.replace("```json", "").replace("```", ""))
        is_good, reason = result.get("is_good", False), result.get("reason", "")
    except Exception:
        is_good, reason = False, "解析异常"
        
    print(f"-> 结果: {is_good} | {reason} (Tokens: {tokens})")
    return {"is_good": is_good, "feedback": reason}

def formatter_node(state: AgentState):
    return {"final_output": f"查询: {state['query']}\n推荐: {len(state['papers'])} 篇"}

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
    return "formatter" if state["is_good"] or state["iteration"] >= 3 else "planner"

workflow.add_conditional_edges("evaluator", route_after_eval)
app = workflow.compile()