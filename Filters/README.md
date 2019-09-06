## run.py



Required arguments:
-i/--input_file INPUT_FILE 
-d/--dataset_name DATASET_NAME 
-c/--creator CREATOR


The *-i/--input_file* should be a text-based file (ie. txt,csv) containing:
- one clean sentence per line (delimited by a newline \n)
- no header
- no index



To run the code:

```
$ python run.py -i INPUT_FILE.* -d "DATASET NAME" -c CREATOR

```


Example:


```
$ python run.py -i ./test_sentences.txt -d "IMDB test" -c Andrea

```


*Filteration Pipeline*
1. Coreference Resolution Checker (keep)
2. Gender Pronoun Checker (keep)
3. Pronoun Link (remove)
4. Human Name Link (remove)
5. Gendered Term Link (remove)
