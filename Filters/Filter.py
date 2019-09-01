from io import open
from allennlp.predictors import Predictor
from allennlp.models.archival import load_archive
import pandas as pd
import pprint
import argparse
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
    return sentences

def dataset_code(code):
    if len(code) != 6:
        raise argparse.ArgumentTypeError("Dataset code must be of length 6")
    if code[2] != "-":
        raise argparse.ArgumentTypeError("Dataset code must include a dash between your name and dataset name")
    if code[:2].upper() not in ["AN", "CA", "IN", "SI", "YA"]:
        raise argparse.ArgumentTypeError("Dataset code must start with the first 2 characters of your name")
    return code

class Filter(object):
    def __init__(self):
        # ALLEN NLP Corereference pre-trained model
        self.predictor = Predictor.from_archive(
            load_archive('https://s3-us-west-2.amazonaws.com/allennlp/models/coref-model-2018.02.05.tar.gz',
                         weights_file=None, overrides=""), 'coreference-resolution')

    def get_dataframe(self, data):

        GENDER_PRONOUNS = ['he', 'she', 'him', 'her', 'his', 'hers', 'himself', 'herself']
        coref_output = []
        gp_output = []
        coref_range = []
        tok_sent = []
        test_gp = []
        no_coref_output = []

        # FILTER #1: Coreference Resolution Checker
        # if coref exists, append 1 to the list called coref_output, else append 0
        # the Gender Pronoun filter below will use this information (it will only check the sentences that have coref)

        for line in tqdm(data):
            coref_line = {"document": line.strip()}
            try:
                coref_json = self.predictor.predict_json(coref_line)
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

        return final_candidates_df

def main():
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--input_file", default=None, type=str, required=True,
                        help="Should be a .csv file, no index, no header, one clean sentence per line.")
    parser.add_argument("--dataset_code", default=None, type=dataset_code, required=True,
                        help="First 2 characters of your name, dash, first 3 characters of the dataset name.")
    args = parser.parse_args()

    data = read_csv(args.input_file)
    df = Filter().get_dataframe(data)
    df.to_csv( args.dataset_code.upper() + "_df", header=True)

if __name__ == '__main__':
    main()







