# LLM-Sentiment

This repo contains the data and code for our paper "Sentiment Analysis in the Era of Large Language Models: A Reality Check".

## Usage
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


## Citation
If the code is used in your research, please star our repo and cite our paper as follows:
```
@misc{zhang2023sentiment,
      title={Sentiment Analysis in the Era of Large Language Models: A Reality Check},
      author={Wenxuan Zhang and Yue Deng and Bing Liu and Sinno Jialin Pan and Lidong Bing},
      year={2023},
      eprint={2305.15005},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```