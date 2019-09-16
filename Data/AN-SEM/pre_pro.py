import re
import os
import pandas as pd
import itertools

directory = './'
folder = os.fsencode(directory)

all_str = []

for file in os.listdir(folder):
	filename = os.fsdecode(file)
	i = 0
	if filename.endswith('.txt'):
		try:
			#f_open = open(directory + '/' + filename , "rb")
			f_open = pd.read_csv(directory + '/' + filename, sep="\n", header=None)
			
			for i in f_open[0]:
				this_str = ''
				try:
					idx = i.index("	")
					this_str = i[:idx]
				except ValueError:
					print("Error with the string: " + i)
				all_str.append(this_str)

		except:
			print("Error with file: " + filename)


all_str = list(dict.fromkeys(all_str))
df = pd.DataFrame(all_str)
df.to_csv('cleaned.csv', index=False, header=None)
