import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from react import run_react, setup_openrouter_client
import json
import time
def exact_match(predicted,gold):
    return predicted.lower() == gold.lower()

def evaluate(data,client,n=100):
    results = []
    correct = 0
    
    for i, item in enumerate(data[:n]):
        question = item["question"]
        gold = item["answer"]
        
        print(f"Question {i+1} : {question}")
        predicted = run_react(question,client)
        em = exact_match(predicted,gold)
        if em :
            correct+=1
        results.append(
            {
                "question": question,
                "gold" : gold,
                "predicted":predicted,
                "em":em
            })
        
        print(f"Predicted : {predicted}")
        print(f"Match : {em}")
        
        with open("eval_results.json", "w") as f:
            json.dump(results, f, indent=2)
            time.sleep(10)
            
    score = correct / n
    print(f"\nFinal EM: {correct}/{n} = {score:.1%}")
    return results

print("Key loaded:", os.environ.get("OPENROUTER_API_KEY") is not None)
openrouter_client = setup_openrouter_client()
with open("eval/hotpot_val.json") as f:
    data = [json.loads(line) for line in f]
print(evaluate(data, openrouter_client, n=100))