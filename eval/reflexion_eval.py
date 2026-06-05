
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reflexion import run_reflexion
from eval.exact_match import exact_match
from react import setup_openrouter_client
import json
import time

def evaluate(data,actor_client,reflect_client,n=1):
    results = []
    correct = 0
    
    for i, item in enumerate(data[:n]):
        question = item["question"]
        gold = item["answer"]
        
        print(f"Question {i+1} : {question}")
        trials,predicted = run_reflexion(question,actor_client,reflect_client,gold)
        if predicted is None:
            predicted =""
        em = exact_match(predicted,gold)
        if em :
            correct+=1
        results.append(
            {
                "question": question,
                "gold" : gold,
                "predicted":predicted,
                "em":em,
                "trials":trials
            })
        
        print(f"Predicted : {predicted}")
        print(f"Match : {em}")
        
        with open("reflexion_eval_results.json", "w") as f:
            json.dump(results, f, indent=2)
        time.sleep(10)
            
    score = correct / n
    print(f"\nFinal EM: {correct}/{n} = {score:.1%}")
    return results

if __name__ == "__main__":
    print("Key loaded:", os.environ.get("OPENROUTER_API_KEY") is not None)
    actor_client = setup_openrouter_client()
    self_reflection_client= setup_openrouter_client()
    with open("eval/hotpot_val.json") as f:
        data = [json.loads(line) for line in f]
    print(evaluate(data, actor_client,self_reflection_client, n=100))