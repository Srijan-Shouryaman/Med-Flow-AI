"""Relevance predicate, per-case scoring, and aggregation."""
from __future__ import annotations
import time
from collections import defaultdict
from statistics import mean
from .golden_set import PATIENT_ID, Case
from .adapter import retrieve

def norm(chunk: dict) -> dict:
    c = dict(chunk)
    c["_text"] = (c.get("text") or "").lower()
    c["_report"] = str(c.get("report_date", ""))[:10]
    return c

def is_relevant(case: Case, chunk: dict) -> bool:
    if chunk.get("chunk_type") not in case.types:
        return False
    if case.reports and chunk["_report"] not in case.reports:
        return False
    if case.terms and not any(t.lower() in chunk["_text"] for t in case.terms):
        return False
    return True

def score_case(case: Case, chunks: list[dict], k: int) -> dict:
    top = chunks[:k]
    rel = [is_relevant(case, c) for c in top]
    hit = any(rel)
    rr = next((1.0/(i+1) for i,r in enumerate(rel) if r), 0.0)
    if case.reports:
        found = {c["_report"] for c,r in zip(top,rel) if r}
        report_recall = len(found.intersection(case.reports))/len(case.reports)
    else:
        report_recall = 1.0 if hit else 0.0
    return {"hit":float(hit), "rr":rr, "precision":sum(rel)/k,
            "report_recall":report_recall, "first_rank":next((i+1 for i,r in enumerate(rel) if r),None)}

def evaluate(cases, mode="rerank", k=5):
    rows=[]
    for case in cases:
        start=time.perf_counter()
        chunks=[norm(c) for c in retrieve(case.query, PATIENT_ID, k=k, mode=mode)]
        rows.append({"case":case,"chunks":chunks,"latency":time.perf_counter()-start,**score_case(case,chunks,k)})
    return rows

def aggregate(rows):
    if not rows: return {}
    lat=sorted(r["latency"] for r in rows)
    return {"n":len(rows),"hit@k":mean(r["hit"] for r in rows),
            "mrr":mean(r["rr"] for r in rows),"precision@k":mean(r["precision"] for r in rows),
            "report_recall@k":mean(r["report_recall"] for r in rows),
            "latency_p50_s":lat[len(lat)//2],"latency_p95_s":lat[min(len(lat)-1,int(.95*len(lat)))]}

def aggregate_by_category(rows):
    groups=defaultdict(list)
    for r in rows: groups[r["case"].category].append(r)
    return {cat:aggregate(group) for cat,group in groups.items()}
