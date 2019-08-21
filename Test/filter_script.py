import csv
import sys
from io import open
from allennlp.predictors import Predictor
from allennlp.models.archival import load_archive
import pandas as pd
import pprint
import re
from tqdm import tqdm
pp = pprint.PrettyPrinter(indent=1)

from filters import filter_by_corpus

# Use the NLTK Downloader to obtain the resources that you need for this script:
import nltk
from nltk.tokenize import sent_tokenize
# nltk.download('averaged_perceptron_tagger')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('maxent_ne_chunker')
# nltk.download('words')
# nltk.download('tagsets')

def read_csv(input_file, quotechar=None):
    """Reads a comma separated value file."""
    with open(input_file, "r", encoding="utf-8-sig") as f:
        text = f.read()
    sentences = sent_tokenize(text)
    sentences = [re.sub('"', '', quotation) for quotation in sentences] #remove quotation marks
    return sentences

def read_tsv(input_file, quotechar=None):
    """Reads a tab separated value file."""
    with open(input_file, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
        lines = []
        for line in reader:
            if sys.version_info[0] == 2:
                line = list(unicode(cell, 'utf-8') for cell in line)
            lines.append(line)
        return lines

GENDER_PRONOUNS = ['he', 'she', 'him', 'her', 'his', 'hers', 'himself', 'herself']
coref_output = []
gp_output = []
coref_range = []
tok_sent = []
test_gp = []
no_coref_output = []

#ALLEN NLP Corereference pre-trained model
predictor = Predictor.from_archive(
    load_archive('https://s3-us-west-2.amazonaws.com/allennlp/models/coref-model-2018.02.05.tar.gz',
                 weights_file=None, overrides=""), 'coreference-resolution')

# Import Data
input_path = "./test_sentences.csv"
data = read_csv(input_path)

# FILTER #1: Coreference Resolution Checker
# if coref exists, append 1 to the list called coref_output, else append 0
# the Gender Pronoun filter below will use this information (it will only check the sentences that have coref)

for line in tqdm(data):
    coref_line = {"document": line.strip()}
    try:
        coref_json = predictor.predict_json(coref_line)
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
        break
    except:
        # print("problem sentence: ", line)
        no_coref_output.append(line)

    coref_range.append(coref_json['clusters'])
    tok_sent.append(coref_json['document'])

    if len(coref_json['clusters']) > 0:
        coref_output.append(1)  # coref cluster exists

    else:
        coref_output.append(0)  # coref cluster does not exist

# FILTER #2: Gender Pronoun Checker
# if the GP exists, append 1 to the list called gp_output, else append 0
# the below filter called "filter_by_corpus" will use this list to keep or reject a sentence
for i in range(0, len(data)):
    if coref_output[i] == 1: # checks is coref exists from list that was built above
        for cluster in coref_range[i]:
            test_gp = []
            if any([((c[0] == c[1]) and (tok_sent[i][c[0]]).lower() in GENDER_PRONOUNS) for c in cluster]):
                test_gp.append(True)
            else:
                test_gp.append(False)

        if any(test_gp):
            gp_output.append(1) # GP exists
        else:
            gp_output.append(0) # GP does not exists

    else:
        gp_output.append(0)  # coref cluster doesn't exists so don't look for gp pronoun

assert (len(data) == len(coref_output) == len(gp_output) == len(coref_range) == len(
    tok_sent)), "DIM OF COREF & GP OUT NOT SAME"

# FILTER #3: Pronoun Link
# if the filter finds a pronoun link, it will remove that sentence.
# ie She was very polite, she knew how to comport herself.
pronoun_link = filter_by_corpus(data, tok_sent, coref_range, gp_output, "pro")

# FILTER #4: Human Name Link
# if the filter finds a human name link, it will remove that sentence.
# ie Josie was very polite, she knew how to comport herself.
human_name = filter_by_corpus(data, tok_sent, coref_range, gp_output, "name")

# FILTER #5: Gendered Term Link
# if the filter finds a gendered term link, it will remove that sentence.
# ie The queen was very polite, she knew how to comport herself.
gendered_term = filter_by_corpus(data, tok_sent, coref_range, gp_output, "term")

# by passing in "all" as a parameter, you call filters 3, 4, and 5 at the same time
final_candidates = filter_by_corpus(data, tok_sent, coref_range, gp_output, "all")

print("FILTER PASSED WITH NO ERROR")

assert (len(data) == len(human_name) == len(final_candidates) == len(gendered_term) == len(
    pronoun_link)), "DIM OF FILTER OUT NOT SAME"

building_df = {'Sentences': data, 'Tok Sentences': tok_sent, 'Coref Ranges': coref_range, 'Coreference': coref_output,
               'Gender pronoun': gp_output, 'Gender link': pronoun_link,'Human Name': human_name,
               'Gendered term': gendered_term, 'Final candidates': final_candidates}

final_candidates_df = pd.DataFrame(building_df)

# write the final candidates sentences (the ones that made it passed the filters) to csv
final_candidates_df[final_candidates_df['Final candidates'] == 1]['Sentences'].to_csv("final_candidate_sentences",
                                                                      header=False, index=True)
# write dataframe to csv
final_candidates_df.to_csv("final_candidate_df", header=True, index=True)





