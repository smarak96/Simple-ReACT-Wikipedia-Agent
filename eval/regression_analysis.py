import json

with open("react_eval_results.json") as f:
    react_results = {r["question"]: r for r in json.load(f)}

with open("reflexion_eval_results.json") as f:
    reflexion_results = {r["question"]: r for r in json.load(f)}

common = set(react_results) & set(reflexion_results)

react_only    = []  # ReAct correct, Reflexion wrong
reflexion_only = []  # Reflexion correct, ReAct wrong
both_correct  = []
both_wrong    = []

for q in common:
    r = react_results[q]
    f = reflexion_results[q]
    if r["em"] and f["em"]:
        both_correct.append(q)
    elif r["em"] and not f["em"]:
        react_only.append(q)
    elif not r["em"] and f["em"]:
        reflexion_only.append(q)
    else:
        both_wrong.append(q)

print(f"Questions compared: {len(common)}")
print()
print(f"Both correct        : {len(both_correct)}")
print(f"Both wrong          : {len(both_wrong)}")
print(f"Reflexion only      : {len(reflexion_only)}  ← reflection gained these")
print(f"ReAct only          : {len(react_only)}   ← reflection lost these (regressions)")
print()

if react_only:
    print("--- Regressions (ReAct correct, Reflexion wrong) ---")
    for q in react_only:
        r = react_results[q]
        f = reflexion_results[q]
        print(f"\n  Q: {q}")
        print(f"  Gold      : {r['gold']}")
        print(f"  ReAct     : {r['predicted']}")
        print(f"  Reflexion : {f['predicted']} (trial {f['trials']})")
