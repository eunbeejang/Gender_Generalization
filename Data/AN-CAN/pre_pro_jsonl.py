import json
import pandas as pd



data =[]
with open('AX-g.jsonl') as jsonl_content:
    read = [json.loads(jline) for jline in jsonl_content]
    [data.extend([i['hypothesis'], i['premise']]) for i in read]


data = list(dict.fromkeys(data))
df = pd.DataFrame(data)
df.to_csv('cleaned.csv', index=False, header=None)