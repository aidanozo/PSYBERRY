import json
import os
from transformers import AutoTokenizer

MAX_TOKEN_LENGTH = 512
RAW_PATH = "../data/raw/mental_health_counseling_conversations.json"
# ADDITIONAL_PATH = "../data/additional/additional_data.json"  # comentat
OUTPUT_PATH = "../data/processed/clean_data.json"


def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_valid_pair(prompt, response, tokenizer):
    if not prompt or not response or prompt.strip() == "" or response.strip() == "":
        return False, 0, 0

    prompt_len = len(tokenizer.encode(prompt))
    response_len = len(tokenizer.encode(response))
    total_len = prompt_len + response_len

    if total_len > MAX_TOKEN_LENGTH:
        return False, prompt_len, response_len

    return True, prompt_len, response_len


def extract_clean_pairs(data, tokenizer, source_name=""):
    total_pairs = 0
    too_long = 0
    empty_issues = 0
    max_prompt_len = 0
    max_response_len = 0
    clean_pairs = []

    for conv in data:
        if len(conv) < 2:
            continue

        for i in range(0, len(conv) - 1, 2):
            prompt = conv[i]
            response = conv[i + 1]
            total_pairs += 1

            valid, prompt_len, response_len = is_valid_pair(prompt, response, tokenizer)

            if not valid:
                if prompt_len == 0 and response_len == 0:
                    empty_issues += 1
                    print(f"[SKIPPED - EMPTY] [{source_name}] Index {i}: Empty prompt or response.")
                elif prompt_len + response_len > MAX_TOKEN_LENGTH:
                    too_long += 1
                    print(f"[SKIPPED - LONG] [{source_name}] {prompt_len + response_len} tokens: '{prompt[:60]}...' -> '{response[:60]}...'\n")
                continue

            max_prompt_len = max(max_prompt_len, prompt_len)
            max_response_len = max(max_response_len, response_len)
            clean_pairs.append([prompt.strip(), response.strip()])

    print(f"\nReport from {source_name or 'unknown source'}:")
    print(f"- Total pairs processed: {total_pairs}")
    print(f"- Empty pairs skipped: {empty_issues}")
    print(f"- Overlength pairs skipped: {too_long}")
    print(f"- Max prompt length: {max_prompt_len}")
    print(f"- Max response length: {max_response_len}")
    print(f"- Clean pairs retained: {len(clean_pairs)}\n")

    return clean_pairs


def main():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
    tokenizer.pad_token = tokenizer.eos_token

    print("Processing raw data...")
    raw_data = load_data(RAW_PATH)
    clean_raw = extract_clean_pairs(raw_data, tokenizer, source_name="RAW")

    # === Comentat: Additional data ===
    # print("Processing additional data...")
    # additional_data = load_data(ADDITIONAL_PATH)
    # clean_additional = extract_clean_pairs(additional_data, tokenizer, source_name="ADDITIONAL")

    # all_clean_data = clean_raw + clean_additional
    all_clean_data = clean_raw  # doar datele de baza

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_clean_data, f, indent=4, ensure_ascii=False)

    print(f"\nCleaned data saved to: {OUTPUT_PATH}")
    print(f"Total cleaned pairs: {len(all_clean_data)}")


if __name__ == "__main__":
    main()
