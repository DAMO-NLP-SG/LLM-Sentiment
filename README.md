# LLM-Sentiment

This repo contains the data and code for our paper "[Sentiment Analysis in the Era of Large Language Models: A Reality Check](https://arxiv.org/abs/2305.15005)".

## Usage
0. fill in your OpenAI api key in the bash files under `script` folder. For example:
```
python predict.py \
--setting zero-shot \
--model chat \
--use_api \
--api #your api here
```

1. Run zero-shot and evaluate
```
bash script/run_zero_shot.sh
bash script/eval_zero_shot.sh
```

2. Run few-shot and evaluate
```
bash script/run_few_shot.sh
bash script/eval_few_shot.sh
```

## Note
1. To view the summary of prompts and evaluation results, please navigate to the output folder and check the respective task folder.
2. You can specify `--selected_tasks` and `--selected_datasets` to only run with certain tasks or datasets.


## Citation
If the code is used in your research, please star our repo and cite our paper as follows:
```
@inproceedings{zhang-etal-2024-sentiment,
    title = "Sentiment Analysis in the Era of Large Language Models: A Reality Check",
    author = "Zhang, Wenxuan  and
      Deng, Yue  and
      Liu, Bing  and
      Pan, Sinno  and
      Bing, Lidong",
    booktitle = "Findings of the Association for Computational Linguistics: NAACL 2024",
    month = jun,
    year = "2024",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-naacl.246/",
    pages = "3881--3906",
}
```
