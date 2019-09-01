## Filter.py

The *input_file* should be a csv and contain:

- one clean sentence per line
- no header
- no index

The *dataset_code* should be the first three characters of the dataset name


Example:
```bash
python Filter.py --input_file ../Test/test_sentences.csv --dataset_code TES
```

1. Coreference Resolution Checker (keep)
2. Gender Pronoun Checker (keep)
3. Pronoun Link (remove)
4. Human Name Link (remove)
5. Gendered Term Link (remove)
