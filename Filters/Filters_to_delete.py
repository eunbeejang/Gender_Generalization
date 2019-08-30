from allennlp.predictors import Predictor
from allennlp.models.archival import load_archive
from nltk.tokenize.treebank import TreebankWordDetokenizer
import pandas as pd
import re
import unidecode
import pprint
from tqdm import tqdm
import nltk
from nltk.tokenize import sent_tokenize
from nltk import ne_chunk, pos_tag
from nltk.tokenize import sent_tokenize, word_tokenize

pp = pprint.PrettyPrinter(indent=1)
from nltk.tokenize.treebank import TreebankWordDetokenizer

# from loaders import load_IMDB, load_gluternberg
from remove import check_remove, filter_by_corpus
from highlight_tagger import insert_tags
from loaders import load_IMDB, load_gutenberg, load_general

# Use the NLTK Downloader to obtain the resources that you need for this script:
# nltk.download('averaged_perceptron_tagger')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('maxent_ne_chunker')
# nltk.download('words')
# nltk.download('tagsets')

""" If you are using a gutenberg dataset, choose the load_gutenberg loader 
    If you are using IMDB dataset, choose the load_IMDB loader
    If you have a clean txt that is ready for testing, use the load_general loader
"""


class Dataloader(object):
    def __init__(self, final_candidates_filename, filter_by_corpus, output_df, encoding="utf-8"):
        self.final_candidates_filename = final_candidates_filename
        self.encoding = encoding
        self.output_df = output_df
        self.filter_by_corpus = filter_by_corpus
        self.predictor = Predictor.from_archive(
            load_archive('https://s3-us-west-2.amazonaws.com/allennlp/models/coref-model-2018.02.05.tar.gz',
                         weights_file=None, overrides=""), 'coreference-resolution')

    # is_data_marked: set to True if you want your final candidate to contain html tags that will highlight the clusters
    def filter_to_file(self, data, is_data_marked):
        coref_true = []

        GENDER_PRONOUNS = ['he', 'she', 'him', 'her', 'his', 'hers', 'himself', 'herself']
        coref_output = []
        gp_output = []
        coref_range = []
        final_sentences = []
        tok_sent = []
        test_gp = []
        no_coref_output = []
        # coref_count = 0

        # check if a sentence has coref link
        for line in tqdm(data[1]):
            coref_line = {"document": line.strip()}
            try:
                coref_json = self.predictor.predict_json(coref_line)
            except KeyboardInterrupt:
                print("KeyboardInterrup")
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

        # check
        for i in range(0, len(data[1])):
            if coref_output[i] == 1:
                for cluster in coref_range[i]:
                    test_gp = []
                    if any([((c[0] == c[1]) and (tok_sent[i][c[0]]).lower() in GENDER_PRONOUNS) for c in cluster]):
                        test_gp.append(True)
                    else:
                        test_gp.append(False)  # gp pronoun exists

                if any(test_gp):
                    gp_output.append(1)
                else:
                    gp_output.append(0)

            else:
                gp_output.append(0)  # coref cluster doesn't exists so don't look for gp pronoun
        print(len(data[1]))
        print(len(coref_output))
        print(len(gp_output))
        print(len(coref_range))
        print(len(tok_sent))
        assert (len(data[1]) == len(coref_output) == len(gp_output) == len(coref_range) == len(
            tok_sent)), "DIM OF COREF & GP OUT NOT SAME"

        print("gp_output length", len(gp_output))

        pronoun_link = self.filter_by_corpus(data[1], tok_sent, coref_range, gp_output, "pro")
        human_name = self.filter_by_corpus(data[1], tok_sent, coref_range, gp_output, "name")
        gendered_term = self.filter_by_corpus(data[1], tok_sent, coref_range, gp_output, "term")
        final_candidates = self.filter_by_corpus(data[1], tok_sent, coref_range, gp_output, "all")

        print("FILTER PASSED WITH NO ERROR")

        print(len(data[1]), len(human_name), len(final_candidates), len(gendered_term), len(pronoun_link))

        assert (len(data[1]) == len(human_name) == len(final_candidates) == len(gendered_term) == len(
            pronoun_link)), "DIM OF FILTER OUT NOT SAME"

        building_df = {'Sentences': data[1], 'index': data[0], 'Corpus': data[2], 'Tok Sentences': tok_sent,
                       'Coref Ranges': coref_range, 'Coreference': coref_output, 'Gender pronoun': gp_output,
                       'Gender link': pronoun_link, 'Human Name': human_name,
                       'Gendered term': gendered_term, 'Final candidates': final_candidates}

        plotting_df = pd.DataFrame(building_df)
        # print(plotting_df.head(5)) #check format

        final_sents = []

        for i in range(0, len(final_candidates)):
            if final_candidates[i] == 1:
                final_sents.append([data[0][i], data[1][i], data[2][i], coref_range[i]])

        df = pd.DataFrame(final_sents)

        # print(df.head(5)) #check format

        # print(len(final_sentence_list), "final sentence list")
        # print(len(final_index), "final index list")

        # print(final_sentence_list)

        # final_df = {'Sentences': final_sentence_list, 'Index': final_index}
        # = pd.DataFrame(final_df)
        # print(plotting_df.head[5])
        # write to csv
        # plotting_df[plotting_df['Final candidates'] == 1]['Sentences'].to_csv(self.final_candidates_filename, header=False, index=None)

        plotting_df.to_csv(self.output_df, header=True, index=None)
        df.to_csv(self.final_candidates_filename, header=True, index=None)
        # final_df2.to_csv(self.final_candidates_filename, header=True, index= None)

        if is_data_marked:
            marked_data = []
            for i in range(0, len(final_candidates)):
                if final_candidates[i] == 1:
                    marked_data.append(
                        TreebankWordDetokenizer().detokenize(insert_tags(tok_sent[i], coref_range[i], GENDER_PRONOUNS)))

            marked_df = pd.DataFrame({'text': marked_data})  # Turk needs a header called 'text'
            marked_df.to_csv(self.final_candidates_filename + "_marked.csv", header=True, index=None)

        return plotting_df, df


# Used temporarily for testing
if __name__ == '__main__':
    input_path = '../datasets/gutenberg/clean_gutenbergs/masterclean_new.csv'  ## set up for gutenberg
    final_candidates_filename = "master_finalcandidates_new"
    output_df = "master_allfilters_df"
    is_data_marked = False  # set to True if you want your final candidate sentences to contain html tags that will highlight the clusters

    dataloader = Dataloader(final_candidates_filename, filter_by_corpus, output_df)
    data = load_gutenberg(input_path)
    dataloader.filter_to_file(data, is_data_marked)