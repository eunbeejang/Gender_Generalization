import re
import pandas as pd
from nltk.tokenize import sent_tokenize


file_path = "data/clean/text"
data = pd.read_csv(file_path, sep="\n", header=None)
data_lst = []

for idx, row in data.iterrows():
	clean = re.sub(r"\t", " ", row[0])
	clean = clean.lstrip('0123456789 ')
	data_lst.append(clean)

data_concat = ' '.join(data_lst)

data_lst = sent_tokenize(data_concat)

df = pd.DataFrame(data_lst)
df.to_csv('cleaned.csv', index=False, header=None)
