import json

banned_words = {"therapist", "counsellor", "psychologist", "psychoanalyst", "therapy"}

with open("additional_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

filtered_data = [
    pair for pair in data
    if not any(word.lower() in (pair[0] + " " + pair[1]).lower() for word in banned_words)
]

with open("additional_data.json", "w", encoding="utf-8") as f:
    json.dump(filtered_data, f, indent=4, ensure_ascii=False)