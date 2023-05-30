import os
import openai
import pandas as pd
import argparse
import random
import requests
from tqdm import tqdm
import numpy as np
from tenacity import (
    retry,
    stop_after_attempt,
    wait_fixed,
)
import concurrent.futures

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers", type=int, default=1, help="Number of samples to use, better under 3")
    parser.add_argument("--setting", type=str, default="zero-shot", help="[zero-shot, few-shot, majority, random]")
    parser.add_argument("--seed", type=int, default=42, help="[0, 1, 42]")
    parser.add_argument("--shots", type=int, default=-1, help="[1, 5, 10]")
    parser.add_argument('--use_api', action='store_true', help='use api or not')
    parser.add_argument("--api", type=str, default=None, help="api key")
    parser.add_argument("--selected_tasks", type=str, default=None, help="list of string of tasks, e.g '[\"sc\"]'")
    parser.add_argument("--selected_datasets", type=str, default=None, help="list of string of datasets")
    parser.add_argument("--ignored_datasets", type=str, default=None, help="list of string of datasets")
    parser.add_argument("--model", type=str, default="chat", help="[chat, flant5, flanul2]")
    parser.add_argument("--skip_runned", action="store_true", help="skip runned dataset")
    return parser.parse_args()


def before_retry_fn(retry_state):
    if retry_state.attempt_number > 1:
        print(f"Retrying API call. Attempt #{retry_state.attempt_number}, f{retry_state}")


def parallel_query_chatgpt_model(args):
    return query_chatgpt_model(*args)


def parallel_query_davinci_model(args):
    return query_davinci_model(*args)


# Function to query the OpenAI model
# @retry(wait=wait_random_exponential(min=10, max=60), stop=stop_after_attempt(6), before=before_retry_fn)
@retry(wait=wait_fixed(10), stop=stop_after_attempt(6), before=before_retry_fn)
def query_chatgpt_model(api_key: str, prompt: str, model: str = "gpt-3.5-turbo", max_tokens: int = 256, temperature: float = 0):
    openai.api_key = api_key
    try:
        completions = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            n=1,
            stop=None,
            temperature=temperature,
        )
        output = completions.choices[0].message.content.strip()
    except Exception as e:
        print(e)
    return output


# @retry(wait=wait_random_exponential(min=10, max=60), stop=stop_after_attempt(6), before=before_retry_fn)
@retry(wait=wait_fixed(10), stop=stop_after_attempt(6), before=before_retry_fn)
def query_davinci_model(api_key: str, prompt: str, model: str = "text-davinci-003", max_tokens: int = 256, temperature: float = 0):
    openai.api_key = api_key
    try:
        completions = openai.Completion.create(
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )
        output = completions.choices[0].text.strip()
    except Exception as e:
        print(e)
    return output


def parallel_query_flant5_model(args):
    return query_flant5(*args)


@retry(wait=wait_fixed(10), stop=stop_after_attempt(6), before=before_retry_fn)
def query_flant5(api_key, prompt):
    model_url = "https://api-inference.huggingface.co/models/google/flan-t5-xxl"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "inputs": f"{prompt}",
        "temperature": 0.0
    }
    try:
        response = requests.post(model_url, headers=headers, json=payload)
        pred = response.json()[0]['generated_text'].strip()
    except Exception as e:
        print(response.json())
    return pred


def parallel_query_flanul2_model(args):
    return query_flanul2(*args)


@retry(wait=wait_fixed(10), stop=stop_after_attempt(6), before=before_retry_fn)
def query_flanul2(api_key, prompt):
    model_url = "https://api-inference.huggingface.co/models/google/flan-ul2"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "inputs": f"{prompt}",
        "temperature": 0.0
    }
    try:
        response = requests.post(model_url, headers=headers, json=payload)
        pred = response.json()[0]['generated_text'].strip()
    except Exception as e:
        print(response.json())
    return pred


# Get label space
def get_label_space(task: str, dataset: str) -> list:
    if task == "sc":
        if "asc" in dataset:
            label_space = ["positive", "negative", "neutral"]
        elif dataset in ["imdb", "yelp2", "mr", "sst2"]:
            label_space = ["positive", "negative"]
        elif dataset in ["twitter"]:
            label_space = ["positive", "negative", "neutral"]
        elif dataset in ["yelp5", "sst5"]:
            label_space = ["very positive", "positive", "neutral", "negative", "very negative"]
        else:
            raise NotImplementedError
    elif task == "mast":
        if dataset == "compsent19":
            label_space = ["better", "worse"]
        elif dataset == "emotion":
            label_space = ["anger", "joy", "optimism", "sadness"]
        elif dataset == "hate":
            label_space = ["hate", "non-hate"]
        elif dataset == "offensive":
            label_space = ["offensive", "non-offensive"]
        elif dataset == "irony":
            label_space = ["irony", "non_irony"]
        elif dataset == "stance":
            label_space = ["none", "against", "favor"]
        elif dataset == "implicit":
            label_space = ["positive", "negative", "neutral"]
        else:
            raise NotImplementedError
    elif task == "absa":
        label_space = ["positive", "neutral", "negative"]
        if "asqp" in dataset:
            cat_space = ['ambience general', 'drinks prices', 'drinks quality', 'drinks style_options', 'food general', 'food prices', 'food quality', 'food style_options', 'location general', 'restaurant general', 'restaurant miscellaneous', 'restaurant prices', 'service general']
            label_space = (sorted(label_space), sorted(cat_space))
            return label_space
    else:
        raise NotImplementedError
    return sorted(label_space)


# Function to get the task name and stance target based on the task and dataset
def get_task_name(task: str, dataset: str) -> str:

    if task == "sc":
        if "asc" in dataset:
            task_name = "aspect sentiment classification"
        else:
            task_name = "sentiment classification"
    elif task == "mast":
        if dataset == "stance":
            task_name = f"stance detection"
        elif dataset in ["emotion", "hate", "irony", "offensive"]:
            task_name = f"{dataset} detection"
        elif dataset == "compsent19":
            task_name = "comparative opinions"
        elif dataset == "implicit":
            task_name = "aspect-based implicit sentiment analysis"
        else:
            raise NotImplementedError
    elif task == "absa":
        if "uabsa" in dataset:
            task_name = "unified aspect-based sentiment analysis"
        elif "aste" in dataset:
            task_name = "aspect sentiment triplet extraction"
        elif "asqp" in dataset:
            task_name = "aspect sentiment quad prediction"
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    return task_name.title()


# Define templates for different tasks and datasets
def generate_template(key, label_space, task_name, **kwargs):
    task_definitions = {
        "cls": "Given the sentence, assign a sentiment label from {label_space}.",
        "asc": "Given the sentence, assign a sentiment label towards \"{target}\" from {label_space}.",
        "stance": "Given the sentence, assign a sentiment label expressed by the author towards \"{target}\" from {label_space}.",
        "erc": "Given a list of conversation, assign a sentiment label from {label_space} to each sentence.",
        "uabsa": "Given the sentence, tag all (aspect, sentiment) pairs. Aspect should be substring of the sentence, and sentiment should be selected from {label_space}.",
        "aste": "Given the sentence, tag all (aspect, opinion, sentiment) triplets. Aspect and opinion should be substring of the sentence, and sentiment should be selected from {label_space}.",
        "asqp": "Given the sentence, tag all (category, aspect, opinion, sentiment) quadruples. Aspect and opinion should be substring of the sentence. Category should be selected from {cat_space}. Sentiment should be selected from {label_space}. Only aspect can be 'NULL', category, opinion and sentiment cannot be 'NULL'.",
        "compsent19": "Given the sentence, compare \"{object1}\" to \"{object2}\", and assign an opinion label from {label_space}.",
        "implicit": "Given the sentence, please infer the sentiment towards the aspect \"{target}\". Please select a sentiment label from {label_space}.",
        "irony": "Given the sentence, please determine wheter or not it contains irony. Assign a sentiment label from {label_space}."
    }

    output_formats = {
        "cls": "Return label only without any other text.",
        "irony": "Return label only without any other text.",
        "asc": "Return label only without any other text.",
        "stance": "Return label only without any other text.",
        "erc": "Return a python list of label string only, and do not return any other text.",
        "uabsa": "If there are no aspect-sentiment pairs, return an empty list. Otherwise return a python list of tuples containing two strings in double quotes. Please return python list only, without any other comments or texts.",
        "aste": "Return a python list of tuples containing three strings in double quotes. Please return python list only, without any other comments or texts.",
        "asqp": "Return a python list of tuples containing four strings in double quotes. Please return python list only, without any other comments or texts.",
        "compsent19": "Return label only without any other text.",
        "implicit": "Return label only without any other text.",
    }

    if key == "stance":
        task_name += " ({target})".format(**kwargs)

    task_definition = task_definitions[key].format(**kwargs, label_space=label_space)
    output_format = output_formats[key]

    return task_name, task_definition, output_format


# generate demos
def generate_fix_demo(train_df, task, dataset):
    tuple_list = []
    if dataset == "compsent19":
        for i, row in train_df.iterrows():
            text = row["text"]
            label = row["label_text"]
            o1, o2, _ = eval(row["tuple"])
            text += f" (compare \"{o1}\" to \"{o2}\")"
            tuple_list.append((text, label))
    elif dataset in ["implicit", "asc_lap14", "asc_rest14"]:
        for i, row in train_df.iterrows():
            text = row["text"]
            label = row["label_text"]
            aspect = row["aspect"]
            text += f" (sentiment towards \"{aspect}\")"
            tuple_list.append((text, label))
    elif dataset == "stance":
        for i, row in train_df.iterrows():
            text = row["text"]
            label = row["label_text"]
            domain = row["domain"]
            text += f" (opinion towards \"{domain}\")"
            tuple_list.append((text, label))
    else:
        sub_df = train_df[['text', 'label_text']]
        tuple_list = [tuple(x) for x in sub_df.to_records(index=False)]
    return tuple_list


# Function to generate prompt for the OpenAI model
def generate_prompt(setting, task, dataset, label_space, row, demo_tuples):
    text = row["text"]
    task_name = get_task_name(task, dataset)

    # Use templates to generate prompt
    if task == "sc":
        if "asc" in dataset:
            task_name, task_definition, output_format = generate_template("asc", label_space, task_name=task_name, target=row["aspect"])
        else:
            task_name, task_definition, output_format  = generate_template("cls", label_space, task_name=task_name)
    elif task == "mast":
        if dataset == "stance":
            task_name, task_definition, output_format = generate_template("stance", label_space, task_name=task_name, target=row["domain"])
        elif dataset in ["irony"]:
            task_name, task_definition, output_format = generate_template("irony", label_space, task_name=task_name)
        elif dataset in ["emotion", "hate", "offensive"]:
            task_name, task_definition, output_format = generate_template("cls", label_space, task_name=task_name)
        elif dataset == "compsent19":
            o1, o2, _ = eval(row["tuple"])
            task_name, task_definition, output_format = generate_template("compsent19", label_space, task_name=task_name, object1=o1, object2=o2)
        elif dataset == "implicit":
            task_name, task_definition, output_format = generate_template("implicit", label_space, task_name=task_name, target=row["aspect"])
        else:
            raise NotImplementedError
    elif task == "absa":
        if "uabsa" in dataset:
            task_name, task_definition, output_format = generate_template("uabsa", label_space, task_name=task_name)
        elif "aste" in dataset:
            task_name, task_definition, output_format = generate_template("aste", label_space, task_name=task_name)
        elif "asqp" in dataset:
            senti_space, cat_space = label_space
            task_name, task_definition, output_format = generate_template("asqp", senti_space, task_name=task_name, cat_space=cat_space)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    if setting == "zero-shot":
        prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nSentence:\n{text}"
    elif setting == "few-shot":
        demo_string = ""
        for tup in demo_tuples:
            demo_string += f"\nSentence:\n{tup[0]}\nLabel:{tup[1]}\n"
        prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\n{demo_string}\nSentence:\n{text}\nLabel:\n"
    else:
        raise NotImplementedError
    return prompt


def generate_fake_data(task, dataset, label_space, row):
    # fake data for dev
    if any(substring in dataset for substring in ["uabsa", "aste", "asqp"]):
        try:
            pred = [random.choice(eval(row["label_text"]))]
        except:
            pred = []
    else:
        pred = str(random.choice(label_space))
    return pred


def process_dataset(task, dataset, file_path, output_folder, api_key, setting, num_workers, train_path, shots, verbose=False, args=None):
    df = pd.read_csv(file_path)

    if setting in ["few-shot", "majority"]:
        train_df = pd.read_csv(train_path)
    else:
        train_df = None

    print(f"Predict on Task: {task}, Dataset: {dataset}")
    label_space = get_label_space(task, dataset)

    predictions = []
    prompts = []

    prompt_args = []
    if setting in ["zero-shot", "random", "majority"]:
        demo_tuples = None
    elif setting == "few-shot":
        demo_tuples = generate_fix_demo(train_df, task, dataset)
    else:
        raise NotImplementedError

    max_len = 0
    if setting in ["zero-shot", "few-shot"]:
        if api_key is not None:
            if args.model == "chat":
                parallel_call = parallel_query_chatgpt_model
            elif args.model == "davinci":
                parallel_call = parallel_query_davinci_model
            elif args.model == "flant5":
                parallel_call = parallel_query_flant5_model
            elif args.model == "flanul2":
                parallel_call = parallel_query_flanul2_model
            else:
                raise NotImplementedError

            for index, row in df.iterrows():
                prompt = generate_prompt(setting, task, dataset, label_space, row, demo_tuples)
                max_len = max(max_len, len(prompt.split()))
                if index == 0:
                    prompt_sample = prompt

                prompt_args.append((api_key, prompt))

            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                predictions = list(tqdm(executor.map(parallel_call, prompt_args), total=len(prompt_args), desc=f"Processing {dataset}"))

            for args in prompt_args:
                prompts.append(args[1])
        else:
            for index, row in tqdm(df.iterrows()):
                prompt = generate_prompt(setting, task, dataset, label_space, row, demo_tuples)
                max_len = max(max_len, len(prompt.split()))
                if index == 0:
                    prompt_sample = prompt
                pred = generate_fake_data(task, dataset, label_space, row)
                prompts.append(prompt)
                predictions.append(pred)
    elif setting in ["random", "majority"]:
            if setting == "majority":
                most_common = train_df["label_text"].value_counts().idxmax()
            for index, row in tqdm(df.iterrows()):
                prompt_sample = ""
                if setting == "random":
                    pred = generate_fake_data(task, dataset, label_space, row)
                elif setting == "majority":
                    # should use train file
                    pred = most_common
                prompts.append("")
                predictions.append(pred)
    else:
        raise NotImplementedError

    # print(f"max_len: {max_len}")
    if verbose:
        print(prompt)
    df["prediction"] = predictions
    df["prompt"] = prompts

    output_path = os.path.join(output_folder, f"prediction.csv")
    df.to_csv(output_path, index=False)

    return prompt_sample


# Function to process the task and process datasets
def process_task(args, task, api_key, selected_datasets=None, ignored_datasets=None):

    setting = args.setting
    num_workers = args.num_workers
    shots = args.shots
    seed = args.seed
    model = args.model

    task_folder = os.path.join("data", f"{task}")

    if setting in ["zero-shot", "random", "majority"]:
        output_task_folder = f"outputs/{setting}/model_{model}/seed_{seed}/{task}"
    elif setting == "few-shot":
        output_task_folder = f"outputs/{setting}/shot_{shots}/model_{model}/seed_{seed}/{task}"
    else:
        raise NotImplementedError

    prompt_samples = []
    dataset_names = []

    def check_entry(entry, selected_datasets, ignored_datasets):
        return entry.is_dir() and (selected_datasets is None or entry.name in selected_datasets) \
            and (ignored_datasets is None or entry.name not in ignored_datasets)

    entries = (entry for entry in sorted(os.scandir(task_folder), key=lambda e: e.name) if check_entry(entry, selected_datasets, ignored_datasets))
    for dataset in entries:
        output_dataset_folder = os.path.join(output_task_folder, dataset.name)
        os.makedirs(output_dataset_folder, exist_ok=True)

        file_path = os.path.join(dataset.path, "test.csv")

        if setting in ["zero-shot", "random"]:
            train_path = None
        elif setting == "majority":
            train_path = os.path.join(f"csv/{task}/{dataset.name}", "train.csv")
        elif setting == "few-shot":
            train_path = os.path.join(dataset.path, f"shot_{shots}", f"seed_{seed}", "train.csv")
        else:
            raise NotImplementedError

        if args.skip_runned:
            pred_file = os.path.join(output_dataset_folder, "prediction.csv")
            if os.path.exists(pred_file):
                print(f"{task} {dataset.name} skiped")
                continue

        prompt_sample = process_dataset(task, dataset.name, file_path, output_dataset_folder, api_key, setting, num_workers, train_path, shots, args=args)

        prompt_samples.append(prompt_sample)
        dataset_names.append(dataset.name)

    prompt_file = os.path.join(output_task_folder, "prompt.txt")
    with open(prompt_file, 'w') as f:
        for task_dataset, prompt in zip(dataset_names, prompt_samples):
            f.write('-'*100+'\n')
            f.write(f"{task}-{task_dataset}:\n{prompt}\n\n")


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    selected_tasks = eval(args.selected_tasks) if args.selected_tasks else ["sc", "mast", "absa"]
    selected_datasets = eval(args.selected_datasets) if args.selected_datasets else None
    ignored_datasets = eval(args.ignored_datasets) if args.ignored_datasets else None

    api_key = args.api

    for task in selected_tasks:
        process_task(args, task, api_key, selected_datasets, ignored_datasets)

if __name__ == "__main__":
    main()