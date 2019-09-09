import re
import pandas as pd




this = pd.read_csv("./WSCollection.xml", sep="\n", header=None, encoding='latin-1')

start = 0

data = []
temp = []
for i in this[0]:
	if start == 1:
		temp.append(re.sub('<.*?>|(&#8226;)', '', i).strip())
	if i == "<text>":
		start = 1
	if i == "</text>":
		start = 0
		data.append(' '.join(temp))
		temp = []



df = pd.DataFrame(data)
df.to_csv('cleaned.csv', index=False, header=None)
