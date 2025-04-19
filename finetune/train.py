import json
import csv
import os
import torch
import gc
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from bert_score import score as bert_score

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
    model.to(device)

    with open("../data/processed/train_data.json", "r", encoding="utf-8") as f:
        unprocessed_train_data = json.load(f)

    train_data = []
    for conversation in unprocessed_train_data:
        for i in range(0, len(conversation) - 1, 2):
            query = conversation[i]
            response = conversation[i + 1]
            train_data.append([query, response])

    train_texts = []
    for pair in train_data:
        text = "User: " + pair[0] + tokenizer.eos_token + "Therapist: " + pair[1] + tokenizer.eos_token
        train_texts.append(text)

    class ConversationDataset(Dataset):
        def __init__(self, texts, tokenizer, max_length=512):
            self.texts = texts
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            text = self.texts[idx]
            encoding = self.tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_length
            )
            input_ids = encoding["input_ids"].squeeze(0)
            labels = input_ids.clone()

            eos_token_id = tokenizer.eos_token_id
            eos_indices = (input_ids == eos_token_id).nonzero(as_tuple=True)

            if len(eos_indices[0]) >= 1:
                second_part_start = eos_indices[0][0].item() + 1
                labels[:second_part_start] = -100
            else:
                labels[:] = -100

            return {"input_ids": input_ids, "labels": labels}

    def collate_fn(batch):
        input_ids = torch.stack([item["input_ids"] for item in batch])
        labels = torch.stack([item["labels"] for item in batch])
        return {"input_ids": input_ids, "labels": labels}

    batch_size = 2
    dataset = ConversationDataset(train_texts, tokenizer, max_length=512)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    epochs = 10
    loss_records = []

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            loss_records.append([epoch + 1, step, loss.item()])
            print(f"Epoch {epoch + 1}, Step {step}, Loss: {loss.item()}")

            torch.cuda.empty_cache()
            gc.collect()

    os.makedirs("results", exist_ok=True)
    with open("results/training_loss.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Epoch", "Step", "Loss"])
        for record in loss_records:
            writer.writerow(record)

    model.save_pretrained("model/dialoGPT")
    tokenizer.save_pretrained("model/dialoGPT")

    # Evaluare pe test_data
    with open("../data/processed/test_data.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)

    model.eval()

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    bleu_scores, rouge1_scores, rougeL_scores, meteor_scores = [], [], [], []
    references_bert, outputs_bert = [], []
    metrics_records = []

    with torch.no_grad():
        for idx, pair in enumerate(test_data):
            query, reference = pair[0], pair[1]
            print("User:", query)
            print("Reference:", reference)

            prompt = f"User: {query}{tokenizer.eos_token}Therapist:"
            input = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
            input_ids = input["input_ids"].to(device)
            attention_mask = input["attention_mask"].to(device)

            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=100,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False
            )

            output_text = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
            output_text = output_text.strip()
            reference = reference.strip()

            bleu = sentence_bleu([reference.split()], output_text.split())
            rouge = scorer.score(reference, output_text)
            meteor = meteor_score([reference.split()], output_text.split())

            bleu_scores.append(bleu)
            rouge1_scores.append(rouge["rouge1"].fmeasure)
            rougeL_scores.append(rouge["rougeL"].fmeasure)
            meteor_scores.append(meteor)
            references_bert.append(reference)
            outputs_bert.append(output_text)

            metrics_records.extend([
                [idx, "BLEU", bleu],
                [idx, "ROUGE-1", rouge["rouge1"].fmeasure],
                [idx, "ROUGE-L", rouge["rougeL"].fmeasure],
                [idx, "METEOR", meteor]
            ])

            print("Bot:", output_text)
            print("-" * 50)

        P, R, F1 = bert_score(outputs_bert, references_bert, lang="en", device=device)
        average_bert_f1 = F1.mean().item()

        for idx, f1 in enumerate(F1):
            metrics_records.append([idx, "BERTScore", f1.item()])

    with open("results/testing_metrics.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["test_pair", "metric", "metric_value"])
        for record in metrics_records:
            writer.writerow(record)

    print("Average BLEU:", sum(bleu_scores) / len(bleu_scores))
    print("Average ROUGE-1:", sum(rouge1_scores) / len(rouge1_scores))
    print("Average ROUGE-L:", sum(rougeL_scores) / len(rougeL_scores))
    print("Average METEOR:", sum(meteor_scores) / len(meteor_scores))
    print("Average BERTScore (F1):", average_bert_f1)

if __name__ == "__main__":
    main()