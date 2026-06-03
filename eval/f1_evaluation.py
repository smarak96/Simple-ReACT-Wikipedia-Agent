import json

def f1_score(predicted, gold):
    pred_tokens = set(predicted.lower().split())
    gold_tokens = set(gold.lower().split())
    common = pred_tokens & gold_tokens
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall    = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)

with open("eval_results.json") as f:
    results = json.load(f)

f1_scores = [f1_score(r["predicted"], r["gold"]) for r in results]
avg_f1 = sum(f1_scores) / len(f1_scores)
print(f"F1: {avg_f1:.1%}")