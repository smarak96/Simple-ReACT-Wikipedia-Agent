import json
from collections import Counter

with open("reflexion_eval_results.json") as f:
    results = json.load(f)

total = len(results)
solved = [r for r in results if r["em"]]
unsolved = [r for r in results if not r["em"]]

trial_counts = Counter(r["trials"] for r in solved)

print(f"Total questions: {total}")
print(f"Solved: {len(solved)} | Unsolved: {len(unsolved)}")
print()
print("--- Trial breakdown (solved questions only) ---")
react_baseline = trial_counts.get(0, 0)
reflection_gains = sum(v for k, v in trial_counts.items() if k > 0)
for trial in sorted(trial_counts):
    label = "trial 0 (ReAct baseline)" if trial == 0 else f"trial {trial} (reflection helped)"
    bar = "#" * trial_counts[trial]
    print(f"  {label}: {trial_counts[trial]:3d}  {bar}")

print()
print(f"  unsolved (all trials failed): {len(unsolved):3d}  {'#' * len(unsolved)}")
print()
print(f"ReAct baseline contribution : {react_baseline}/{total} = {react_baseline/total:.1%}")
print(f"Reflection extra gain       : {reflection_gains}/{total} = {reflection_gains/total:.1%}")
