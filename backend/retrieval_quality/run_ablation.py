"""Run sparse/dense/hybrid/rerank ablation. From backend root:
python -m retrieval_quality.run_ablation --k 5
"""
import argparse, json
from pathlib import Path
from .golden_set import CASES, PATIENT_ID
from .metrics import aggregate, aggregate_by_category, evaluate

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--k",type=int,default=5)
    args=parser.parse_args()
    if args.k < 1: parser.error("--k must be >= 1")
    overall,by_category={},{}
    for mode in ("sparse","dense","hybrid","rerank"):
        rows=evaluate(CASES,mode=mode,k=args.k)
        overall[mode]=aggregate(rows)
        by_category[mode]=aggregate_by_category(rows)
    print(f"\\nPatient {PATIENT_ID} | {len(CASES)} golden queries | k={args.k}")
    print(f"{'mode':<9}{'Hit@k':>9}{'MRR':>9}{'P@k':>9}{'RepRec@k':>11}{'p50(s)':>10}{'p95(s)':>10}")
    for mode,m in overall.items():
        print(f"{mode:<9}{m['hit@k']:>9.3f}{m['mrr']:>9.3f}{m['precision@k']:>9.3f}{m['report_recall@k']:>11.3f}{m['latency_p50_s']:>10.2f}{m['latency_p95_s']:>10.2f}")
    cats=sorted(by_category["rerank"])
    print(f"\\nHit@{args.k} by category")
    print(f"{'category':<14}"+"".join(f"{m:>10}" for m in ("sparse","dense","hybrid","rerank")))
    for cat in cats:
        print(f"{cat:<14}"+"".join(f"{by_category[m][cat]['hit@k']:>10.2f}" for m in ("sparse","dense","hybrid","rerank")))
    output={"patient_id":PATIENT_ID,"k":args.k,"query_count":len(CASES),"overall":overall,"by_category":by_category}
    Path("ablation_results.json").write_text(json.dumps(output,indent=2),encoding="utf-8")
    print("\\nSaved ablation_results.json")
if __name__=="__main__": main()
