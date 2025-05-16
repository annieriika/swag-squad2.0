"""
Utilities for extractive QA
"""

from torch.utils.data import DataLoader
import torch
import numpy as np
from transformers import EvalPrediction
from utils_qa import postprocess_qa_predictions


# Preprocessing function for the train and validation datasets from https://github.com/huggingface/transformers/blob/main/examples/pytorch/question-answering/run_qa.py
def preprocess_and_tokenize(train_dataset, dev_dataset, test_dataset, tokenizer, max_seq_length=384, doc_stride=128):
    def prepare_train_features(examples):
        examples["question"] = [q.lstrip() for q in examples["question"]]  # Remove leading spaces
        tokenized_examples = tokenizer(
            examples["question"], examples["context"],
            truncation="only_second",
            max_length=max_seq_length,
            stride=doc_stride,
            return_overflowing_tokens=True, # for long documents
            return_offsets_mapping=True, # identify start and end tokens in original text
            padding="max_length"
        )
        
        # Label processing
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized_examples.pop("offset_mapping")
        
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []
        
        for i, offsets in enumerate(offset_mapping):
            # label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id) if tokenizer.cls_token_id in input_ids else 0

            sequence_ids = tokenized_examples.sequence_ids(i)
            sample_index = sample_mapping[i]
            answers = examples["answers"][sample_index]
            
            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if tokenizer.padding_side == "right" else 0):
                    token_start_index += 1

                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if tokenizer.padding_side == "right" else 0):
                    token_end_index -= 1

                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples

    def prepare_validation_features(examples):
        examples["question"] = [q.lstrip() for q in examples["question"]]  # Remove leading spaces
        tokenized_examples = tokenizer(
            examples["question"], examples["context"],
            truncation="only_second",
            max_length=max_seq_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length"
        )
        
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if tokenizer.padding_side == "right" else 0

            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples

    # Apply the preprocessing for the training set
    tokenized_train = train_dataset.map(
        prepare_train_features,
        batched=True,
        remove_columns=train_dataset.column_names
    )
    
    # Apply the preprocessing for the validation set
    tokenized_dev = dev_dataset.map(
        prepare_validation_features,
        batched=True,
        remove_columns=dev_dataset.column_names
    )

    # Apply the preprocessing for the test set
    tokenized_test = test_dataset.map(
        prepare_validation_features,
        batched=True,
        remove_columns=test_dataset.column_names
    )

    return tokenized_train, tokenized_dev, tokenized_test


def collate_fn(batch):
    """
    Collate function for Hugging Face Dataset.
    Converts a list of dictionaries into batched tensors.
    """
    input_ids = torch.tensor([item['input_ids'] for item in batch], dtype=torch.long)
    attention_mask = torch.tensor([item['attention_mask'] for item in batch], dtype=torch.long)
    token_type_ids = torch.tensor([item['token_type_ids'] for item in batch], dtype=torch.long) if 'token_type_ids' in batch[0] else None
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'token_type_ids': token_type_ids
    }


def make_predictions(model, tokenized_test, batch_size=32):
    model.eval()  
    all_start_logits = []
    all_end_logits = []

    dataloader = DataLoader(tokenized_test, batch_size=batch_size, collate_fn=collate_fn)

    for batch in dataloader:
        input_ids = batch['input_ids'].to(model.device)
        attention_mask = batch['attention_mask'].to(model.device)
        token_type_ids = batch['token_type_ids'].to(model.device) if batch['token_type_ids'] is not None else None
        
        # Run the model on the batch
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        all_start_logits.append(outputs.start_logits.cpu().numpy())
        all_end_logits.append(outputs.end_logits.cpu().numpy())

    # Concatenate all logits across batches
    start_logits = np.concatenate(all_start_logits, axis=0)
    end_logits = np.concatenate(all_end_logits, axis=0)

    return start_logits, end_logits


def make_predictions_with_swa(model, tokenized_test, batch_size=32):
    model.eval()  
    all_start_logits = []
    all_end_logits = []

    dataloader = DataLoader(tokenized_test, batch_size=batch_size, collate_fn=collate_fn)

    model.sample_parameters(scale=0, cov=True)

    for batch in dataloader:
        input_ids = batch['input_ids'].to(model.device)
        attention_mask = batch['attention_mask'].to(model.device)
        token_type_ids = batch['token_type_ids'].to(model.device) if batch['token_type_ids'] is not None else None
        
        # Run the model on the batch
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        all_start_logits.append(outputs.start_logits.cpu().numpy())
        all_end_logits.append(outputs.end_logits.cpu().numpy())

    # Concatenate all logits across batches
    start_logits = np.concatenate(all_start_logits, axis=0)
    end_logits = np.concatenate(all_end_logits, axis=0)

    return start_logits, end_logits


def post_processing_function(examples, features, predictions, stage="eval"):
    # Post-process to match the logits to answers
    predictions = postprocess_qa_predictions(
        examples=examples,
        features=features,
        predictions=predictions,
        version_2_with_negative=True,
        n_best_size=20,
        max_answer_length=30,
        null_score_diff_threshold=0.0,
        output_dir="./results",
        prefix=stage,
    )
    
    formatted_predictions = [
                {"id": str(k), "prediction_text": v, "no_answer_probability": 0.0} for k, v in predictions.items()
            ]

    references = [{"id": str(ex["id"]), "answers": ex["answers"]} for ex in examples]

    return EvalPrediction(predictions=formatted_predictions, label_ids=references)
