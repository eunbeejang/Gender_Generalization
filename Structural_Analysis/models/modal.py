import re

def get_modal(constTree): # Analyze modals by their functions

    MODAL_DICT = {'can' : ['ability', 'suggestion', 'request'],
                'could' : ['ability', 'request', 'suggestion'],
                'may' : ['possibility', 'request', 'permission'],
                'might' :['possibility'],
                'must' : ['obligation', 'certainty'],
                'should' : ['opinion', 'advice'],
                'ought' : ['opinion', 'advice']}

    # These below are considered as modals when they are followed by 'to'                    
    VP_MODAL_DICT = {'able to' : ['ability'],
                'have to' : ['obligation'],'has to' : ['obligation'],'had to' : ['obligation'],
                'need to' : ['obligation'],'needs to' : ['obligation'],'needed to' : ['obligation']}
                

    VP = [i[1] for i in constTree[0] if i[1][0] == 'VP']


    # List all MD tags in all VPs
    modal_found = [[re.sub(r'[\(\)]', '',j) for j in i if re.sub('\(', '',j) == 'MD'] for i in VP]

    # List all MODAL+'to' in all VPs
    # Convert to str first cus str search more efficient than list search
    word_str = [' '.join(map(str, k)) for k in [[re.sub(r'[\(\)]', '',j) 
                for j in i if re.sub(r'[\(\)]', '',j).islower()] for i in VP]] 
    VP_modal_found = [[i.lower() for i in VP_MODAL_DICT.keys() if i.lower() in sub_lst] for sub_lst in word_str]


    # Extract actual word from MD list
    MD = [i[1][1].lower() for i in constTree[0] if i[1][0] == 'MD']
    # Store all possible functions for the word
    MD_intersect = list((set(MD)).intersection(set(MODAL_DICT))) 

    # Check for words in MODAL_DICT
    if (any(modal_found) and any((set(MD)).intersection(set(MODAL_DICT))) and len(MD_intersect) != 0):
        try:
            return MODAL_DICT[list((set(MD)).intersection(set(MODAL_DICT)))[0]]
        except:
            return []

    # Check for words in VP_MODAL_DICT
    elif (not(any(modal_found)) and (any(VP_modal_found))):
        try:
            return VP_MODAL_DICT[VP_modal_found[0][0]]
        except:
            return VP_modal_found[0]
    else: return []

