from datasets import load_dataset

ds = load_dataset("hotpotqa/hotpot_qa", "fullwiki")
validation_hotpot = ds['validation']
validation_hotpot.to_json("eval/hotpot_val.json")
