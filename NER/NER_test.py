import os

os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
os.environ['WANDB_DISABLED'] = 'true'
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from datasets import load_dataset, load_metric, Dataset, DatasetDict
import numpy as np
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report, accuracy_score
import sys

def read_dataset_file(file_path):
    with open(file_path, "r", encoding='UTF-8') as f:
        content = f.read().strip()
        sentences = content.split("\n\n")
        data = []
        for sentence in sentences:
            tokens = sentence.split("\n")
            token_data = []
            for token in tokens:
                token_data.append(token.split())
            data.append(token_data)
    return data


test_data = read_dataset_file("test/test_set.txt")


def convert_to_dataset(data, label_map):
    formatted_data = {"tokens": [], "ner_tags": []}
    for sentence in data:
        tokens = [token_data[0] for token_data in sentence]
        ner_tags = [label_map[token_data[1]] for token_data in sentence]
        formatted_data["tokens"].append(tokens)
        formatted_data["ner_tags"].append(ner_tags)
    return Dataset.from_dict(formatted_data)

label_list = sorted(list(set([token_data[1] for sentence in test_data for token_data in sentence])))
label_map = {label: i for i, label in enumerate(label_list)}

test_dataset = convert_to_dataset(test_data, label_map)

datasets = DatasetDict({
    "test": test_dataset,
})
model_name = sys.argv[1]
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(label_list))


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True, padding=True
    )
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


tokenized_datasets = datasets.map(tokenize_and_align_labels, batched=True)


def compute_metrics(eval_prediction):
    predictions, labels = eval_prediction
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    precision = precision_score(true_labels, true_predictions)
    recall = recall_score(true_labels, true_predictions)
    f1 = f1_score(true_labels, true_predictions)
    accuracy = accuracy_score(true_labels, true_predictions)

    tmp_true_predictions = true_predictions
    tmp_true_labels = true_labels
    tmp_tokens = tokenized_datasets["test"]['tokens']

    for el in tmp_true_predictions:
        el.append('SEP')
    for el in tmp_tokens:
        el.append('SEP')
    for el in tmp_true_labels:
        el.append('SEP')

    df = pd.DataFrame({'token': tmp_tokens, 'truth': true_labels, 'pred': true_predictions})

    df = df.explode(['token', 'truth', 'pred'])

    df1 = df
    df1["token"] = np.where(df["pred"] == "MISC", df["token"] + ' [cit]', df["token"])
    df['id'] = df.index

    df1 = df1.groupby(['id'])['token'].apply(list).reset_index()
    df1['token'] = df1.token.apply(lambda x: ' '.join([str(i) for i in x]))

    nerOutput = open(
        "predictionNer.txt",
        'w', encoding='UTF-8')
    

    for index, row in df1.iterrows():
        output = row['token'].replace('SEP', '')
        nerOutput.write(output + '\n')


    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy
    }


training_args = TrainingArguments(
    output_dir="./test-ner",
    evaluation_strategy="epoch",
    num_train_epochs=8,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_steps=1000,
    learning_rate=5e-5,
    metric_for_best_model="f1",
)


def data_collator(data):
    input_ids = [torch.tensor(item["input_ids"]) for item in data]
    attention_mask = [torch.tensor(item["attention_mask"]) for item in data]
    labels = [torch.tensor(item["labels"]) for item in data]

    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

outputs = trainer.predict(tokenized_datasets["test"])
print(outputs.metrics)
