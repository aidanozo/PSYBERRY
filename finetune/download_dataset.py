import json
from datasets import load_dataset

def main():

    dataset = load_dataset("Amod/mental_health_counseling_conversations")

    data_pairs = []
    for item in dataset["train"]:
        context = item["Context"]
        response = item["Response"]
        data_pairs.append([context, response])

    output_filename = "../data/raw/mental_health_counseling_conversations.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(data_pairs, f, indent=4, ensure_ascii=False)

    print(f"Am salvat {len(data_pairs)} perechi in fisierul '{output_filename}'.")

if __name__ == "__main__":
    main()