

def get_adv(constTree, coref): # Counts the number of adverbs in each sent
	return len([(constTree[1]['pos_tags'][i], coref['document'][i]) 
		for i in range(0, len(constTree[1]['pos_tags'])) 
		if constTree[1]['pos_tags'][i] == "RB" 
		or constTree[1]['pos_tags'][i] == "RBR" 
		or constTree[1]['pos_tags'][i] == "RBS"])