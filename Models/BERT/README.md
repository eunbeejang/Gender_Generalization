To fine-tune BERT on our dataset, run the below command at this location `~/Gender_Generalization/Models/pytorch-pretrained-BERT/examples`

The BERT model has a linear layer on top of the pooled output for sequence classification.
``` 
python run_classifier.py \
  --task_name SST-2 \
  --do_train \
  --do_eval \
  --data_dir ../../../Data/BERT_data/ \
  --bert_model bert-base-cased \
  --max_seq_length 512 \
  --train_batch_size 8 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir ../../../Models/BERT/results/

```
