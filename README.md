# Event-Centric Opinion Mining
- An implementation for ECO v1: Towards Event-Centric Opinion Mining.
- Please contact @Ruoxi Xu (ruoxi2021@iscas.ac.cn) for questions and suggestions.

## Requirements
General
- Python 3.7.10
- CUDA Version 10.1

Python Packages
- see requirements.txt

```python
conda create -n CCL_TASK6 python=3.7
conda activate CCL_TASK6

pip install -r requirements.txt

```

## Quick Start
### Data Format

[comment]: <> (**Additional Statement：** We organize [an evaluation]&#40;http://e-com.ac.cn/ccl2022.html/&#41; in CCL2022. So the annoation in test data is not available currently. After the evaluation, we'll open full dataset.)

Data folder: data/ECOB-ZH .

Before training models, you should first download [data](http://123.57.148.143:9876/down/mRVUaxtM7oUz) and unzip them as follows. 
Data enhance: use Baidu Translate API. First translate dataset from zh to en, and then translate dataset back from en to zh (dataset includes both train and dev). 
```
data
├── ECOB-ZH  
├── ── train.doc.json
├── ── train.ann.json
├── ── dev.doc.json
├── ── dev.ann.json
├── ── test.doc.json
├── ── enhance_train.doc.json
├── ── enhance_train.ann.json
├── ── enhance_dev.doc.json
├── ── enhance_dev.ann.json
├── ── enhance_train_dev.doc.json
└── ── enhance_train_dev.ann.json
```

The data format is as follows:

In train/dev/test.doc.json, each JSON instance represents a document.
```
{
    "Descriptor": {
        "event_id": (int) event_id,
        "text": "Event descriptor."
    },
    "Doc": {
        "doc_id": (int) doc_id,
        "title": "Title of document.",
        "content": [
            {
                "sent_idx": 0,
                "sent_text": "Raw text of the first sentence."
            },
            {
                "sent_idx": 1,
                "sent_text": "Raw text of the second sentence."
            },
            ...
            {
                "sent_idx": n-1,
                "sent_text": "Raw text of the (n-1)th sentence."
            }
        ]
    }
}
```

In train/dev/test.ann.json, each JSON instance represents an opinion extracted from documents.
```
[
	{
            "event_id": (int) event_id,
            "doc_id": (int) doc_id,
            "start_sent_idx": (int) "Sent idx of first sentence of the opinion.",
            "end_sent_idx": (int) "Sent idx of last sentence of the opinion.",
            "argument": (str) "Event argument (opinion target) of the opinion."
  	}
]
```

### Model Train
If unable to auto download, you need to search and download these pretrained model：
- hfl/chinese-macbert-base
- chinese_pretrain_mrc_roberta_wwm_ext_large
- ltp base (word segment model)

#### Step 1: Event-Oriented Opinion Extraction
 (Train and Test) Predict test file during train, and results will be saved at model_combine/task1_combine_data/

- seq_base (3 models, including 0 cross valid)

```python
python eoe_model/seq_base/main.py \
    --train_file train \
    --gradient_accumulation_steps 2 \
    --num_epochs 8 \
    --model_folder seq_train_accum2_epoch8 \
    --test_file test \
    --lr 5e-4 \
    --backbone_lr 1e-6 \
    --batch_size 1 \
    --bert hfl/chinese-macbert-base \
    --data_dir data/ECOB-ZH/ \
    --model_dir model_files/chinese_model/ \
    --result_dir model_combine/task1_combine_data/ 
```

```python
python eoe_model/seq_base/main.py \
    --train_file enhance_train \
    --gradient_accumulation_steps 2 \
    --num_epochs 5 \
    --model_folder seq_enhance_train_accum2_epoch5 \
    --test_file test \
    --lr 5e-4 \
    --backbone_lr 1e-6 \
    --batch_size 1 \
    --bert hfl/chinese-macbert-base \
    --data_dir data/ECOB-ZH/ \
    --model_dir model_files/chinese_model/ \
    --result_dir model_combine/task1_combine_data/
```

```python
python eoe_model/seq_base/main.py \
    --train_file enhance_train_dev \
    --gradient_accumulation_steps 2 \
    --num_epochs 8 \
    --model_folder seq_enhance_train_dev_accum2_epoch8 \
    --test_file test \
    --lr 5e-4 \
    --backbone_lr 1e-6 \
    --batch_size 1 \
    --bert hfl/chinese-macbert-base \
    --data_dir data/ECOB-ZH/ \
    --model_dir model_files/chinese_model/ \
    --result_dir model_combine/task1_combine_data/
```

- attention_base (3 models, including 3 Cross-validation)
```python
python eoe_model/attention_base/main.py \
       --hidden_dim 768 \
       --dropout 0.4 \
       --lr 0.0005 \
       --bert hfl/chinese-macbert-base \
       --model_folder attention_cat_0.4_768_5fold \
       --batch_size 1 \
       --num_epochs 8 \
       --k_fold 5 \
       --data_dir data/ECOB-ZH/ \
       --result_dir model_combine/task1_combine_data/
```

```python
python eoe_model/attention_base/main.py \
       --hidden_dim 512 \
       --dropout 0.4 \
       --lr 0.001 \
       --bert hfl/chinese-macbert-base \
       --model_folder attention_cat_0.4_512_10fold \
       --batch_size 1 \
       --num_epochs 8 \
       --k_fold 10 \
       --data_dir data/ECOB-ZH/ \
       --result_dir model_combine/task1_combine_data/
```

```python
python eoe_model/attention_base/main.py \
       --hidden_dim 512 \
       --dropout 0.5 \
       --lr 0.001 \
       --bert hfl/chinese-macbert-base \
       --model_folder attention_cat_0.5_512_10fold \
       --batch_size 1 \
       --num_epochs 8 \
       --k_fold 10 \
       --data_dir data/ECOB-ZH/ \
       --result_dir model_combine/task1_combine_data/
```


####  Step 2: Opinion Target Extraction
(Train) Predict dev file rather than test file
- mrc (4 models, including 1 Cross-validation)
```python
python ote_model/mrc/main.py \
    --word_seg_model jieba \
    --num_train_epochs 6 \
    --overwrite_cache \
    --do_train \
    --train_file train \
    --predict_file dev \
    --gradient_accumulation_steps 3 \
    --language chinese \
    --model_name_or_path pretrained_model/chinese_pretrain_mrc_roberta_wwm_ext_large \
    --do_eval \
    --do_lower_case \
    --learning_rate 3e-5 \
    --per_gpu_eval_batch_size=32 \
    --per_gpu_train_batch_size=2 \
    --evaluate_during_training \
    --output_dir model_files/chinese_model/mrc_jieba_train_accum3_epoch6/ \
    --data_dir data/ECOB-ZH/ \
    --result_dir result/chinese_result
```

```python
python ote_model/mrc/main.py \
    --word_seg_model ltp \
    --num_train_epochs 5 \
    --overwrite_cache \
    --do_train \
    --train_file train \
    --predict_file dev \
    --gradient_accumulation_steps 3 \
    --language chinese \
    --model_name_or_path pretrained_model/chinese_pretrain_mrc_roberta_wwm_ext_large \
    --do_eval \
    --do_lower_case \
    --learning_rate 3e-5 \
    --per_gpu_eval_batch_size=32 \
    --per_gpu_train_batch_size=2 \
    --evaluate_during_training \
    --output_dir model_files/chinese_model/mrc_ltp_train_accum3_epoch5/ \
    --data_dir data/ECOB-ZH/ \
    --result_dir result/chinese_result
```

```python
python ote_model/mrc/main.py \
    --word_seg_model ltp \
    --num_train_epochs 6 \
    --overwrite_cache \
    --do_train \
    --train_file enhance_train \
    --predict_file dev \
    --gradient_accumulation_steps 3 \
    --language chinese \
    --model_name_or_path pretrained_model/chinese_pretrain_mrc_roberta_wwm_ext_large \
    --do_eval \
    --do_lower_case \
    --learning_rate 3e-5 \
    --per_gpu_eval_batch_size=32 \
    --per_gpu_train_batch_size=2 \
    --evaluate_during_training \
    --output_dir model_files/chinese_model/mrc_ltp_enhance_train_accum3_epoch6/ \
    --data_dir data/ECOB-ZH/ \
    --result_dir result/chinese_result
```

```python
python ote_model/mrc/main.py \
    --k_fold 10 \
    --word_seg_model jieba \
    --num_train_epochs 5 \
    --overwrite_cache \
    --do_train \
    --train_file train \
    --predict_file dev \
    --gradient_accumulation_steps 3 \
    --language chinese \
    --model_name_or_path pretrained_model/chinese_pretrain_mrc_roberta_wwm_ext_large \
    --do_eval \
    --do_lower_case \
    --learning_rate 3e-5 \
    --per_gpu_eval_batch_size=32 \
    --per_gpu_train_batch_size=2 \
    --evaluate_during_training \
    --output_dir model_files/chinese_model/mrc_jieba_train_accum3_epoch5_cross_valid/ \
    --data_dir data/ECOB-ZH/ \
    --result_dir result/chinese_result
```

### Model Test and Model Combine

####  Step 1: eoe model combine (combine 4 times)
```python
python model_combine/task1_model_combine.py \
    --use_default_list \
    --select_cnt 5 \
    --output_file attention_cat_0.4_512_combine.pred.json \
    --combine_file_list attention_k_fold_list_1
```

```python
python model_combine/task1_model_combine.py \
    --use_default_list \
    --select_cnt 6 \
    --output_file attention_cat_0.5_512_combine.pred.json \
    --combine_file_list attention_k_fold_list_2
```

```python
python model_combine/task1_model_combine.py \
    --use_default_list \
    --select_cnt 3 \
    --output_file attention_cat_0.4_768_combine.pred.json \
    --combine_file_list attention_k_fold_list_3
```

```python
python model_combine/task1_model_combine.py \
    --use_default_list \
    --select_cnt 3 \
    --output_file combine_test.ann.json \
    --combine_file_list final_combine_list
```

####  Step 2: generate test.ann.json
Padding each argument with an empty string, file will be saved at data/ECOB-ZH/test.ann.json
```python
python arg_padding.py \
    --input_file model_combine/task1_combine_data/combine_test.ann.json \
    --output_file data/ECOB-ZH/test.ann.json
```
####  Step 3: mrc test (4 times)
File will be saved at model_combine/task2_combine_data/
```python
python ote_model/mrc/main.py \
    --result_file mrc_jieba_train_accum3_epoch6_best.ann.json \
    --word_seg_model jieba \
    --num_train_epochs 6 \
    --overwrite_cache \
    --train_file train \
    --predict_file test \
    --gradient_accumulation_steps 3 \
    --language chinese \
    --model_name_or_path pretrained_model/chinese_pretrain_mrc_roberta_wwm_ext_large \
    --do_eval \
    --do_lower_case \
    --learning_rate 3e-5 \
    --per_gpu_eval_batch_size=32 \
    --per_gpu_train_batch_size=2 \
    --evaluate_during_training \
    --output_dir model_files/chinese_model/mrc_jieba_train_accum3_epoch6/ \
    --data_dir data/ECOB-ZH/ \
    --result_dir model_combine/task2_combine_data
```

```python
python ote_model/mrc/main.py \
    --result_file mrc_ltp_train_accum3_epoch5_best.ann.json \
    --word_seg_model ltp \
    --num_train_epochs 5 \
    --overwrite_cache \
    --train_file train \
    --predict_file test \
    --gradient_accumulation_steps 3 \
    --language chinese \
    --model_name_or_path pretrained_model/chinese_pretrain_mrc_roberta_wwm_ext_large \
    --do_eval \
    --do_lower_case \
    --learning_rate 3e-5 \
    --per_gpu_eval_batch_size=32 \
    --per_gpu_train_batch_size=2 \
    --evaluate_during_training \
    --output_dir model_files/chinese_model/mrc_ltp_train_accum3_epoch5/ \
    --data_dir data/ECOB-ZH/ \
    --result_dir model_combine/task2_combine_data
```

```python
python ote_model/mrc/main.py \
    --result_file mrc_ltp_enhance_train_accum3_epoch6_best.ann.json \
    --word_seg_model ltp \
    --num_train_epochs 6 \
    --overwrite_cache \
    --train_file enhance_train \
    --predict_file test \
    --gradient_accumulation_steps 3 \
    --language chinese \
    --model_name_or_path pretrained_model/chinese_pretrain_mrc_roberta_wwm_ext_large \
    --do_eval \
    --do_lower_case \
    --learning_rate 3e-5 \
    --per_gpu_eval_batch_size=32 \
    --per_gpu_train_batch_size=2 \
    --evaluate_during_training \
    --output_dir model_files/chinese_model/mrc_ltp_enhance_train_accum3_epoch6/ \
    --data_dir data/ECOB-ZH/ \
    --result_dir model_combine/task2_combine_data
```

```python
python ote_model/mrc/main.py \
    --k_fold 10 \
    --word_seg_model jieba \
    --num_train_epochs 5 \
    --overwrite_cache \
    --train_file train \
    --predict_file test \
    --gradient_accumulation_steps 3 \
    --language chinese \
    --model_name_or_path pretrained_model/chinese_pretrain_mrc_roberta_wwm_ext_large \
    --do_eval \
    --do_lower_case \
    --learning_rate 3e-5 \
    --per_gpu_eval_batch_size=32 \
    --per_gpu_train_batch_size=2 \
    --evaluate_during_training \
    --output_dir model_files/chinese_model/mrc_jieba_train_accum3_epoch5_cross_valid/ \
    --data_dir data/ECOB-ZH/ \
    --result_dir model_combine/task2_combine_data
```

####  Step 4: ote model combine (1 time)
Final predict result will be saved at model_combine/task2_combine_data/team_g9OrQLtH.ann.json
```python
python model_combine/task2_model_combine.py \
    --use_default_list \
    --output_file team_g9OrQLtH.ann.json \
    --combine_file_list final_combine_list
```
## License
