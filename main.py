import os
import time
import json
import requests
from openai import OpenAI

# ================= 1. 配置区 =================
# 智谱 API Key（一开始是直接写死在这里测试，但现在是用环境变量了）
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")

# 初始化智谱客户端（智谱兼容 OpenAI 接口格式）
client = OpenAI(
    api_key=ZHIPU_API_KEY,
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)

# ================= 2. 检索模块（OpenAlex 免费替代版） =================
def search_papers(query: str, limit: int = 10):
    url = "https://api.openalex.org/works"
    params = {
        "search": query,
        "per-page": limit,
        "mailto": "xiuzhe.26@intl.zju.edu.cn"  # 建议填写真实邮箱，自动进入免费不限流的礼貌池
    }
    
    print(f"正在通过 OpenAlex 检索: '{query}' ...")
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()
    
    results = []
    for item in data.get("results", []):
        # 解析摘要
        abstract_inverted = item.get("abstract_inverted_index")
        abstract = ""
        if abstract_inverted:
            words = []
            for word, positions in abstract_inverted.items():
                for pos in positions:
                    words.append((pos, word))
            abstract = " ".join([w[1] for w in sorted(words)])

        # 安全获取 venue 名称，防止 None 报错
        try:
            location = item.get("primary_location") or {}
            source = location.get("source") or {}
            venue_name = source.get("display_name", "未知")
        except Exception:
            venue_name = "未知"

        results.append({
            "title": item.get("title") or "无标题",
            "abstract": abstract,
            "year": item.get("publication_year"),
            "venue": venue_name,
            "url": item.get("id")
        })
    return results

# ================= 3. 排序模块 =================
def rank_papers(query: str, papers: list):
    if not papers:
        return []

    # 把论文拼成文本供大模型阅读
    paper_texts = ""
    for i, p in enumerate(papers):
        title = p.get('title', '无标题')
        abstract = p.get('abstract') or '无摘要'
        paper_texts += f"[{i}] 标题: {title}\n摘要: {abstract[:400]}...\n\n"

    prompt = f"""你是一个学术搜索排序助手。用户的查询是："{query}"

以下是候选论文：
{paper_texts}

请根据与用户查询的相关性，从高到低排序。只输出一个 JSON 数组，格式为：
[{{"index": 论文编号, "score": 0-10的整数, "reason": "一句话理由"}}]
不要输出任何其他多余的内容（不要Markdown标记）。"""

    print("正在调用大模型进行相关性排序...")
    response = client.chat.completions.create(
        model="glm-4-flash", # 智谱免费模型
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )
    
    content = response.choices[0].message.content.strip()
    # 清理可能的 markdown 标记
    if content.startswith("```json"):
        content = content[7:-3]
    elif content.startswith("```"):
        content = content[3:-3]
        
    try:
        return json.loads(content.strip())
    except Exception as e:
        print(f"解析 LLM 输出失败，原始输出为：\n{content}")
        return []

# ================= 4. 主流程 =================
if __name__ == "__main__":
    test_query = "用强化学习解决大模型幻觉的论文"
    
    # 1. 检索
    raw_papers = search_papers(test_query, limit=10)
    print(f"检索到 {len(raw_papers)} 篇论文，准备排序...\n")
    
    # 2. 排序
    ranked_results = rank_papers(test_query, raw_papers)
    
    # 3. 输出
    if ranked_results:
        print("====== 最终排序结果 ======")
        # 按分数从高到低排
        for item in sorted(ranked_results, key=lambda x: x.get("score", 0), reverse=True):
            idx = item.get("index")
            if 0 <= idx < len(raw_papers):
                paper = raw_papers[idx]
                print(f"[{item.get('score')}/10] {paper.get('title')}")
                print(f"   理由: {item.get('reason')}")
                print(f"   年份: {paper.get('year')} | 会议: {paper.get('venue')}\n")