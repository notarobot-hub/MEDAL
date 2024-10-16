# MEDAL

Official Implementation of Better Late Than Never: Model-Agnostic Hallucination Post-Processing Framework Towards Clinical Text Summarization

## Environment

```shell
conda create -n medal python=3.8
pip install -r requirements.txt
```

## Download Datasets

- HQS: https://github.com/abachaa/MEDIQA2021/tree/main/Task1
- RRS: https://github.com/abachaa/MEDIQA2021/tree/main/Task3
- ACI-Bench: https://github.com/StanfordMIMI/clin-summ/tree/main/data/d2n

## The Medical Infilling Model

```shell
# training dataset generation
python preprocess/gen_mask_train.py

# inference dataset generation
python preprocess/gen_mask_test.py

# train
sh infilling_model/mask_run.sh

# non-factual summaries generation
sh infilling_model/mask_pred.sh
```


## The Hallucination Correction Model

Config files with hyper-parameters are available in json files under the "./config"


```shell
# train
python main --config ./config/config-file --do_train

# predict
python main --config ./config/config-file --do_test
```

## Metrics
We provide the code necessary to obtain the data files used in computing metrics.
```shell
# ALL_medical_term_file.txt and ALL_medical_term_map.json
python preprocess/gen_medical_term_collection.py

# test_link.json
python preprocess/entity_link.py
```

```shell
cd metrics

# calculate the baseline metrics
/bin/bash metric_run.sh

# calculate the metrics after MEDAL's correction
/bin/bash corr_metric_run.sh
```