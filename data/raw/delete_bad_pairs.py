import json
import re

banned_words = {"therapist", "counsellor", "psychologist", "psychoanalyst", "therapy"}

allowed_pattern = re.compile(r"^[\w\s.,!?'\-:;()\"%&@/]+$", re.UNICODE)

bad_substrings = {"\r\n\r\n", "\n\n", "\r\r"}

with open("mental_health_counseling_conversations.json", "r", encoding="utf-8") as f:
    data = json.load(f)

filtered_data = []

for pair in data:
    q, a = pair[0], pair[1]
    full_text = (q + " " + a).lower()

    contains_banned_word = any(word in full_text for word in banned_words)
    contains_bad_substring = any(substr in full_text for substr in bad_substrings)
    contains_illegal_chars = not allowed_pattern.match(q) or not allowed_pattern.match(a)

    if not (contains_banned_word or contains_bad_substring or contains_illegal_chars):
        filtered_data.append([q.strip(), a.strip()])

with open("mental_health_counseling_conversations.json", "w", encoding="utf-8") as f:
    json.dump(filtered_data, f, indent=4, ensure_ascii=False)
