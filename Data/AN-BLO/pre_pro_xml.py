import re
import os
import pandas as pd
import itertools
from nltk.tokenize import sent_tokenize

directory = './'
folder = os.fsencode(directory)

all_str = []

for file in os.listdir(folder):
	filename = os.fsdecode(file)
	i = 0
	if filename.endswith('.xml'):
		try:
			#f_open = open(directory + '/' + filename , "rb")
			f_open = pd.read_csv(directory + '/' + filename, sep="\n", header=None, encoding='utf-8')
			idx = [i for i in range(len(f_open)) if f_open[0][i] == "<post>"][0]+1
			f_str = sent_tokenize(f_open[0][idx])
			new_str = [re.sub('<.*?>|(&#8226;)', '', i).strip() for i in f_str if len(i) > 50 and i[0].isupper() and "." in i]
	#			f_str = re.sub('[^a-zA-Z\.\,\'\"]+', '', f_str)
			all_str = itertools.chain(all_str, new_str)

		except:
			print("Error with file: " + filename)


df = pd.DataFrame(all_str)
df.to_csv('cleaned.csv', index=False, header=None)
