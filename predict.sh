#!/usr/bin/env bash

python3 bert/run_classifier.py --data_dir=./bert/fine-tune_data --bert_config_file=./bert/model/bert_config.json --task_name=cola --vocab_file=./bert/model/vocab.txt --output_dir=./bert/bert_output --init_checkpoint=gs://zpp-bucket-1920/bert-bucket-golkarolka/bert_model/clean_data_tuning/model.ckpt-8939 --do_lower_case=False --max_seq_length=512 --do_predict=True --use_tpu=True --tpu_name=zpp-holist1 --tpu_zone=us-central1-f --num_tpu_cores 8 --save_checkpoints_steps 10000
