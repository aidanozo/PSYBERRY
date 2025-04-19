import json

with open("mental_health_counseling_conversations.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Numar total de perechi: {len(data)}")