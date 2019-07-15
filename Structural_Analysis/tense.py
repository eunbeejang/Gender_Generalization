import re

def get_tense(constTree):

    TENSE_DICT = {'VB':'base',
                'VBD':'past tense',
                'VBG':'present participle',
                'VBN':'past participle',
                'VBP':'present tense non-3rd.sg',
                'VBZ':'present tense 3rd.sg'}

    # Future tense needs a separate lst because 'will' and 'shall' tagged as modals
    FUTURE_DICT = {'will':'future tense', 'shall':'future tense'} 
                
    VP = [i[1] for i in constTree[0] if i[1][0] == 'VP'] # Extract only the verb phrases from the tree

    try: # Check if tense tag exists
        return (TENSE_DICT[[[re.sub('\(', '',j) for j in i if re.sub('\(', '',j) in TENSE_DICT][0] for i in VP][0]])      
    except:
        try: # Check if the future tag exists
            MD = [i[1][1].lower() for i in constTree[0] if i[1][0] == 'MD']
            return FUTURE_DICT[MD[0]]
        except:
            return '' # The sentence contains no tense information