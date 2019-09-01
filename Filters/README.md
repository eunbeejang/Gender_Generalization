## Filter.py

The *input_file* should be a csv and contain:

- one clean sentence per line
- no header
- no index

The *dataset_code* should match the code found in the dataset tracker (first 2 characters of your name, dash, first 3 characters of the dataset name)


Example:
```bash
python Filter.py --input_file ../Test/test_sentences.csv --dataset_code CA-TES
```

1. Coreference Resolution Checker (keep)
2. Gender Pronoun Checker (keep)
3. Pronoun Link (remove)
4. Human Name Link (remove)
5. Gendered Term Link (remove)
