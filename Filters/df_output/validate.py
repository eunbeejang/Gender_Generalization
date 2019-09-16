import pandas as pd
import random
import os
import argparse

def random_validate(file_path, num_random_out):
	# Loading validate_master
	master_df = 'validate_master.csv'
	if not os.path.exists(master_df):
		validate_master = pd.DataFrame({})
	else:
		validate_master = pd.read_csv(master_df)

	# Fetch only the final candidates
	data = pd.read_csv(file_path)
	final_cand = data.loc[data['Final candidates'] == 1.0]
	final_cand = pd.concat([final_cand['Data ID'], final_cand['Sentences']], axis=1, sort=False)
	idx = final_cand.index

	# Randomly select num_random_out number of sentences
	random_id = []
	while len(random_id) < int(num_random_out):
	    random_id.append(random.choice(idx))
	    random_id = list(dict.fromkeys(random_id))

	df = final_cand.loc[random_id]


	# Concatenate validate_master with current df containing randomly fetched final candidate sentences
	validate_master_out = df.append(validate_master)

	validate_master_out.to_csv(master_df, index=None, header=['Data ID', 'Sentences'], encoding='utf-8-sig')

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_file', required=True,
                        help="Name of the input file")
    parser.add_argument('-n', '--num_rand_sents', required=True,
                        help="Number of randomly fetched sentences")
    args = parser.parse_args()
    random_validate(args.input_file, args.num_rand_sents)


if __name__ == '__main__':
    main()



