import argparse
import torch
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForTokenClassification, TrainingArguments, Trainer, get_scheduler, \
    AutoModelForTokenClassification
from torch.utils.data import DataLoader
from torch.optim import AdamW
import evaluate


# There are moke tokens than labels, so we need to align them (CLF and POS tokens)
# We can create an artificial label for the special tokens, -100, since it is ignored by the CE loss
def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX
            if label % 2 == 1:
                label += 1
            new_labels.append(label)

    return new_labels


# Tokenize and align labels for all samples of the dataset
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )
    all_labels = examples["ner_tags"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument(
        "--task",
        dest="task",
        type=str,
        help="Whatn task to perform. options are [NER, POS, CHUNK]",
        required=True,
    )


    args = parser.parse_args()

    torch.manual_seed(42)

    raw_datasets = load_dataset("conll2003", trust_remote_code=True)

    print('Dataset summary from HF')
    print(raw_datasets)

    label_names = raw_datasets["train"].features["ner_tags"].feature.names

    print('Loading BERT model...')
    model_checkpoint = "bert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    tokenized_datasets = raw_datasets.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
    )


    id2label = {i: label for i, label in enumerate(label_names)}
    label2id = {v: k for k, v in id2label.items()}

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    train_dataloader = DataLoader(
        tokenized_datasets["train"],
        shuffle=True,
        collate_fn=data_collator,
        batch_size=8,
    )
    eval_dataloader = DataLoader(
        tokenized_datasets["validation"], collate_fn=data_collator, batch_size=8
    )

    model = AutoModelForTokenClassification.from_pretrained(
        model_checkpoint,
        id2label=id2label,
        label2id=label2id,
    )

    optimizer = AdamW(model.parameters(), lr=2e-5)

    num_train_epochs = 3
    num_update_steps_per_epoch = len(train_dataloader)
    num_training_steps = num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    metric = evaluate.load("seqeval")

    progress_bar = tqdm(range(num_training_steps))

    for epoch in range(num_train_epochs):
        # Training
        model.train()
        for batch in train_dataloader:
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

        # Evaluation
        model.eval()
        for batch in eval_dataloader:
            with torch.no_grad():
                outputs = model(**batch)

            predictions = outputs.logits.argmax(dim=-1)
            labels = batch["labels"]

            # Necessary to pad predictions and labels for being gathered
            print(predictions)
            adada
            predictions = accelerator.pad_across_processes(predictions, dim=1, pad_index=-100)
            labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)

            predictions_gathered = accelerator.gather(predictions)
            labels_gathered = accelerator.gather(labels)

            true_predictions, true_labels = postprocess(predictions_gathered, labels_gathered)
            metric.add_batch(predictions=true_predictions, references=true_labels)

        results = metric.compute()
        print(
            f"epoch {epoch}:",
            {
                key: results[f"overall_{key}"]
                for key in ["precision", "recall", "f1", "accuracy"]
            },
        )

