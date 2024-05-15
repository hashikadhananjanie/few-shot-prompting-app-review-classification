"""
1.	2% data traditional fine-tuning & prompt-tuning  - T5 model - Maalej dataset

"""

from transformers import AdamW, T5Tokenizer, T5ForSequenceClassification
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd

from openprompt.data_utils import InputExample
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate
from openprompt.prompts import ManualVerbalizer

from openprompt import PromptForClassification
from openprompt import PromptDataLoader
import torch
import numpy as np
from sklearn.metrics import classification_report
import contractions
import re


def expand_and_clean(text):
    # Expand contractions and clean text
    expanded_text = contractions.fix(text)
    cleaned_text = re.sub(r'[^\w\s\d]', '', expanded_text)  # Remove special characters
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # Remove extra whitespaces
    cleaned_text = cleaned_text.encode('ascii', 'ignore').decode('ascii')  # Remove emojis
    return cleaned_text.strip()


def download_maalej_dataset():
    df = pd.read_csv('maalej-dataset.csv')
    df = df.rename(columns={"review": "text", "task": "label"})
    return df


def preprocess_data(data):
    data['text'] = data['text'].apply(expand_and_clean)

    # Map label strings to integers
    label_mapping_str_int = {"FR": 0, "PD": 1, "RT": 2, "UE": 3}
    data['label'] = data['label'].map(label_mapping_str_int)
    return data


def prepare_datasets_traditional_ft(tokenizer_ml, train_texts, train_labels, val_texts, val_labels, test_texts,
                                    test_labels):
    train_dataset = prepare_dataset_traditional_ft(tokenizer_ml, train_texts, train_labels)
    val_dataset = prepare_dataset_traditional_ft(tokenizer_ml, val_texts, val_labels)
    test_dataset = prepare_dataset_traditional_ft(tokenizer_ml, test_texts, test_labels)
    return train_dataset, val_dataset, test_dataset


def prepare_dataset_traditional_ft(tokenizer_ml, texts, labels):
    encodings = tokenizer_ml(list(texts), padding=True, truncation=True, max_length=256)
    dataset = TensorDataset(
        torch.tensor(encodings['input_ids']),
        torch.tensor(encodings['attention_mask']),
        torch.tensor(labels.tolist())
    )
    return dataset


def prepare_datasets_prompt_tuning(train_texts, train_labels, val_texts, val_labels, test_texts, test_labels):
    dataset = {'train': [], 'validation': [], 'test': []}
    prepare_dataset_prompt_tuning(dataset['train'], train_texts, train_labels)
    prepare_dataset_prompt_tuning(dataset['validation'], val_texts, val_labels)
    prepare_dataset_prompt_tuning(dataset['test'], test_texts, test_labels)
    return dataset


def prepare_dataset_prompt_tuning(dataset, texts, labels):
    for index, (text, label) in enumerate(zip(texts, labels)):
        dataset.append(
            InputExample(
                guid=index,
                text_a=text,
                label=label
            )
        )


def traditional_fine_tuning(train_text, val_text, test_text, train_label, val_label, test_label):

    # Load the pre-trained T5 tokenizer and model
    tokenizer_ml = T5Tokenizer.from_pretrained('t5-base')
    model = T5ForSequenceClassification.from_pretrained('t5-base', num_labels=4)

    # Prepare datasets
    train_dataset, val_dataset, test_dataset = prepare_datasets_traditional_ft(tokenizer_ml, train_text, train_label,
                                                                               val_text, val_label, test_text,
                                                                               test_label)

    # Fine-tune the T5 model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    best_val_loss = float('inf')
    patience = 3
    num_epochs_without_improvement = 0

    for epoch in range(10):  # Train for a maximum of 10 epochs
        model.train()
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids, attention_mask, labels = [dt.to(device) for dt in batch]
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

        # Validate the model
        model.eval()
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, labels = [dt.to(device) for dt in batch]
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()
        val_loss /= len(val_loader)
        print(f'Epoch {epoch + 1}, Validation Loss: {val_loss}')

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            num_epochs_without_improvement = 0
            # Save the model if validation loss improves
            torch.save(model.state_dict(), 'best_model_tr.pt')
        else:
            num_epochs_without_improvement += 1
            if num_epochs_without_improvement == patience:
                print(f'Validation loss did not improve for {patience} epochs. Early stopping.')
                break

    # Load the best model
    model.load_state_dict(torch.load('best_model_tr.pt'))

    # Evaluate the fine-tuned model on the test set
    model.eval()
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    test_preds, test_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, labels = [dt.to(device) for dt in batch]
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            test_preds.extend(preds.cpu().numpy().tolist())
            test_labels.extend(labels.cpu().numpy().tolist())

    # Calculate precision, recall, and F1 scores
    label_mapping = {0: "FR", 1: "PD", 2: "RT", 3: "UE"}
    test_labels_actual = [label_mapping[label] for label in test_labels]
    test_preds_actual = [label_mapping[label] for label in test_preds]
    test_report = classification_report(test_labels_actual, test_preds_actual, digits=4)
    print("------------ Test Report for traditional fine-tuning - Maalej ------------")
    print(test_report)
    df = pd.DataFrame({'review': test_text, 'predicted label': test_preds_actual, 'actual label': test_labels_actual})
    df.to_excel('_TR_PREDICTIONS.xlsx')


def few_shot_prompt_tuning(train_texts, val_texts, test_texts, train_labels, val_labels, test_labels):

    # Define dataset
    dataset = prepare_datasets_prompt_tuning(train_texts, train_labels, val_texts, val_labels, test_texts, test_labels)

    # Load PLM
    plm, tokenizer, model_config, WrapperClass = load_plm("t5", "t5-base")

    # Construct Template
    template_text = '{"placeholder":"text_a"} Classify this review: {"mask"}'
    mytemplate = ManualTemplate(tokenizer=tokenizer, text=template_text)

    # Define Verbalizer
    classes = ["FR", "PD", "RT", "UE"]
    myverbalizer = ManualVerbalizer(tokenizer=tokenizer, classes=classes,
                                    label_words={
                                         "FR": ['feature', 'add', 'wish', 'improve', 'lack', 'miss', 'need',
                                                'suggest'],
                                         "PD": ['freeze', 'fix', 'bug', 'error', 'crash', 'stuck', 'issue', 'problem',
                                                'fail'],
                                         "RT": ['best', 'useful', 'love', 'awesome', 'fantastic', 'excellent',
                                                'rubbish', 'useless', 'wow', 'superb', 'addict', 'nice'],
                                         "UE": ['ui', 'easy', 'graphic']
                                     })

    # Prompt model
    prompt_model = PromptForClassification(plm=plm, template=mytemplate, verbalizer=myverbalizer, freeze_plm=False)

    train_dataloader = PromptDataLoader(dataset=dataset["train"], template=mytemplate, tokenizer=tokenizer,
                                        tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=3,
                                        batch_size=8, shuffle=True, teacher_forcing=False, predict_eos_token=False,
                                        truncate_method="head")

    validation_dataloader = PromptDataLoader(dataset=dataset["validation"], template=mytemplate, tokenizer=tokenizer,
                                             tokenizer_wrapper_class=WrapperClass, max_seq_length=256,
                                             decoder_max_length=3, batch_size=8, shuffle=False, teacher_forcing=False,
                                             predict_eos_token=False, truncate_method="head")

    test_dataloader = PromptDataLoader(dataset=dataset["test"], template=mytemplate, tokenizer=tokenizer,
                                       tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=3,
                                       batch_size=8, shuffle=False, teacher_forcing=False, predict_eos_token=False,
                                       truncate_method="head")

    # Training
    use_cuda = True
    if use_cuda:
        prompt_model.cuda()

    optimizer = AdamW(prompt_model.parameters(), lr=1e-4)
    loss_func = torch.nn.CrossEntropyLoss()

    best_val_loss = np.inf
    patience = 3
    num_epochs_without_improvement = 0

    for epoch in range(10):
        tot_loss = 0
        for step, inputs in enumerate(train_dataloader):
            if use_cuda:
                inputs = inputs.cuda()
            logits = prompt_model(inputs)
            labels = inputs['label']
            loss = loss_func(logits, labels)
            loss.backward()
            tot_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()

        # Validate
        prompt_model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs in validation_dataloader:
                if use_cuda:
                    inputs = inputs.cuda()
                logits = prompt_model(inputs)
                labels = inputs['label']
                val_loss += loss_func(logits, labels).item()
            val_loss /= len(validation_dataloader)

        print(f'Epoch {epoch + 1}, Validation Loss: {val_loss}')

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            num_epochs_without_improvement = 0
            torch.save(prompt_model.state_dict(), 'best_model_pr.pt')
        else:
            num_epochs_without_improvement += 1
            if num_epochs_without_improvement == patience:
                print(f'Validation loss did not improve for {patience} epochs. Early stopping.')
                break

    # Load the best model
    prompt_model.load_state_dict(torch.load('best_model_pr.pt'))

    # Evaluate the prompt-tuned model on the test set
    all_preds = []
    all_labels = []
    cols = ['review', 'predicted label', 'actual label']
    lst = []
    for inputs in test_dataloader:
        if use_cuda:
            inputs = inputs.cuda()
        logits = prompt_model(inputs)
        labels = inputs['label']
        preds = torch.argmax(logits, dim=-1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

        i = 0
        for pred, actual in zip(preds, labels):
            lst.append(
                [tokenizer.decode(inputs['input_ids'][i], skip_special_tokens=True), classes[pred], classes[actual]])
            i = i + 1

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Print the classification report
    report = classification_report(all_labels, all_preds, target_names=classes, digits=4)
    print("------------ Test Report for few-shot prompt-tuning - Maalej ------------")
    print(report)
    df = pd.DataFrame(lst, columns=cols)
    df.to_excel('_PR_PREDICTIONS.xlsx')


def main():
    # Download data
    data = download_maalej_dataset()
    data = preprocess_data(data)

    # Split the data into train, validation, and test sets
    temp_texts, test_texts, temp_labels, test_labels = train_test_split(data['text'], data['label'],
                                                                        random_state=2022,
                                                                        test_size=0.15,
                                                                        stratify=data['label'])

    temp_train_texts, temp_val_texts, temp_train_labels, temp_val_labels = train_test_split(temp_texts, temp_labels,
                                                                                            random_state=2022,
                                                                                            train_size=2584,
                                                                                            test_size=553,
                                                                                            stratify=temp_labels)

    train_texts, _, train_labels, _ = train_test_split(temp_train_texts, temp_train_labels,
                                                       random_state=2022,
                                                       train_size=0.02,
                                                       stratify=temp_train_labels)

    val_texts, _, val_labels, _ = train_test_split(temp_val_texts, temp_val_labels,
                                                   random_state=2022,
                                                   train_size=0.02,
                                                   stratify=temp_val_labels)

    # Set seed
    torch.manual_seed(2022)
    torch.cuda.manual_seed(2022)

    print('-------- Maalej 2% - T5 --------')
    print('-------- dataset sizes ----------')
    print('dataset size: ' + str(len(data)))
    print('train size: ' + str(train_texts.size))
    print('validation size: ' + str(val_texts.size))
    print('test size: ' + str(test_texts.size))

    traditional_fine_tuning(train_texts, val_texts, test_texts, train_labels, val_labels, test_labels)
    few_shot_prompt_tuning(train_texts, val_texts, test_texts, train_labels, val_labels, test_labels)


if __name__ == "__main__":
    main()
