import os
import json
import time
import glob
import difflib
from tqdm import tqdm
from agent_graph import app, TRACE_LOG, llm

def load_local_dataset(data_dir="./pasa-data/RealScholarQuery"):
    file_paths = glob.glob(os.path.join(data_dir, "*.jsonl")) + glob.glob(os.path.join(data_dir, "*.json"))
    dataset = []
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip(): dataset.append(json.loads(line))
    return dataset

def clean_title(s):
    if not s: return ""
    s = s.lower()
    s = ''.join(c if c.isalnum() or c.isspace() else ' ' for c in s)
    return ' '.join(s.split())

def is_match(retrieved_title, gt_title, threshold=0.7):
    r_clean = clean_title(retrieved_title)
    g_clean = clean_title(gt_title)
    if r_clean == g_clean: return True
    if len(g_clean) > 15 and g_clean in r_clean: return True
    if len(r_clean) > 15 and r_clean in g_clean: return True
    return difflib.SequenceMatcher(None, r_clean, g_clean).ratio() >= threshold

def llm_judge_relevance(query, gt_title, retrieved_titles):
    """让大模型判断：标准答案在语义上，是否出现在了候选列表中"""
    texts = "\n".join([f"[{i}] {t}" for i, t in enumerate(retrieved_titles)])
    prompt = f"""用户的查询意图：'{query}'
标准答案的论文标题是：'{gt_title}'
我们检索到了以下论文：
{texts}

请判断：检索列表中的论文，是否有任意一篇在语义上指的是同一篇论文（或者高度相关）？
如果列表里有，请输出 JSON: {{"is_retrieved": true, "index": 对应的编号}}
如果列表里完全没有，请输出 JSON: {{"is_retrieved": false, "index": -1}}
严格只输出 JSON。"""
    
    content, _ = llm.chat(prompt, temperature=0.1)
    try:
        res = json.loads(content.replace("```json", "").replace("```", ""))
        return res.get("is_retrieved", False)
    except:
        return False

def evaluate_one_query(agent_app, query: str, ground_truth_titles: list):
    initial_state = {"query": query, "search_terms": [], "papers": [], "iteration": 0, "is_good": False, "feedback": "", "final_output": ""}
    TRACE_LOG.clear()

    try:
        result = agent_app.invoke(initial_state)
    except Exception as e:
        return 0.0, [], [], 0, 0

    retrieved_papers = result.get("papers", [])
    retrieved_titles = [p.get("title", "") for p in retrieved_papers]
    
    # 1. 计算 Recall (严格基于标题模糊匹配)
    hits = 0
    for gt in ground_truth_titles:
        if any(is_match(rt, gt) for rt in retrieved_titles):
            hits += 1
            
    recall = hits / len(ground_truth_titles) if ground_truth_titles else 0.0
    
    # 2. 智能归因分析 (基于 LLM Judge)
    failure_type = "SUCCESS" if recall > 0 else "RETRIEVAL_FAILURE"
    if recall == 0:
        # 取前两个标准答案，让大模型检查是否搜到了语义相关的论文
        for gt_title in ground_truth_titles[:2]:
            if llm_judge_relevance(query, gt_title, retrieved_titles):
                failure_type = "RANKING_FAILURE"
                break

    tokens = sum(log.get("tokens", 0) for log in TRACE_LOG)
    latency = sum(log.get("latency(s)", 0) for log in TRACE_LOG)
    
    return recall, retrieved_papers, ground_truth_titles, tokens, round(latency, 2), failure_type

if __name__ == "__main__":
    dataset = load_local_dataset("./pasa-data/RealScholarQuery")
    dataset = dataset[:3]  # 先跑 3 条
    
    all_recalls = []
    report_data = []

    for idx, item in enumerate(tqdm(dataset, desc="评测进度")):
        query, gt_titles = item.get("question", ""), item.get("answer", [])
        if not query or not gt_titles: continue

        recall, retrieved, gt, tokens, latency, failure_type = evaluate_one_query(app, query, gt_titles)
        all_recalls.append(recall)
        
        report_data.append({
            "query": query,
            "recall": round(recall, 4),
            "failure_analysis": failure_type,
            "tokens": tokens,
            "latency_s": latency,
            "retrieved_titles": [p["title"] for p in retrieved],
            "ground_truth": gt
        })

    print("\n\n" + "=" * 60)
    print(f"🏆 平均 Recall@20: {sum(all_recalls) / len(all_recalls):.2%}")
    print(f"📊 归因分析: {sum(1 for r in report_data if r['failure_analysis'] == 'RETRIEVAL_FAILURE')} 次检索失败, "
          f"{sum(1 for r in report_data if r['failure_analysis'] == 'RANKING_FAILURE')} 次排序失败")
    print("=" * 60)

    with open("eval_report.json", "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)
    print("\n✅ 详细评测与归因报告已保存至 eval_report.json")