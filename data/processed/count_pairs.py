import json

with open("clean_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Numar total de perechi: {len(data)}")