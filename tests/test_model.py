import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from google import genai
from google.genai import types
import os
import requests
from datetime import datetime
from edit_profile.update import update_profile_from_session
from edit_profile.extract import extract_user_profile
from init_profile import count_filled_profile_fields, start_profile_initialization_session

def build_session_context(session, limit=5):
    context_lines = []
    for user, therapist in session[-limit:]:
        context_lines.append(f"Pacient: {user}")
        context_lines.append(f"Terapeut: {therapist}")
    return "\n".join(context_lines)

def main():
    translator_key = os.environ["PSYBERRY_DEEPL_API_KEY"]
    gemini_key = os.environ["PSYBERRY_GEMINI_API_KEY"]

    client = genai.Client(api_key=gemini_key)
    gemini_model = "models/gemini-1.5-pro-latest"

    translate_url = "https://api.deepl.com/v2/translate"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dialogpt_model = AutoModelForCausalLM.from_pretrained("../finetune/model/dialoGPT")
    dialogpt_model.to(device)
    dialogpt_model.eval()

    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.eos_token = tokenizer.eos_token

    print("Terapeutul este gata sa te asculte. Scrie 'exit' ca sa iesi.\n")
    user_profile_text = extract_user_profile()

    session = []

    if count_filled_profile_fields() < 5:
        start_profile_initialization_session()
        user_profile_text = extract_user_profile()

    while True:
        query = input("Tu: ").strip()
        raw_input = query

        data_query_en = {
            "auth_key": translator_key,
            "text": query,
            "source_lang": "RO",
            "target_lang": "EN"
        }
        query = requests.post(translate_url, data=data_query_en).json()["translations"][0]["text"]

        if raw_input.lower() in ["La revedere", "Pa", "Ramas bun", "exit"]:

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"session_{timestamp}.json"

            with open(filename, "w", encoding="utf-8") as f:
                json.dump(session, f, ensure_ascii=False, indent=2)

            print(f"\nConversatia a fost salvata in '{filename}'")

            update_profile_from_session(session_folder=".")

            print("Terapeutul: Ne auzim data viitoare. Ai grija de tine!")
            break

        prompt = (
            "You are a friendly and concise therapist. Consider the following user profile when answering.\n\n"
            f"User Profile:\n{user_profile_text}\n\n"
            "Answer shortly and empathetically.\n"
        )
        full_input = prompt + "\nPatient: " + query + "\nTherapist:"

        encoding = tokenizer(
            full_input + tokenizer.eos_token,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        output_ids = dialogpt_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=100,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.9,
            temperature=0.7,
            repetition_penalty=1.5
        )

        output_text = tokenizer.decode(output_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

        data_query_ro = {
            "auth_key": translator_key,
            "text": query,
            "source_lang": "EN",
            "target_lang": "RO"
        }

        data_output_text = {
            "auth_key": translator_key,
            "text": output_text,
            "source_lang": "EN",
            "target_lang": "RO"
        }

        query = requests.post(translate_url, data=data_query_ro).json()["translations"][0]["text"]
        output_text = requests.post(translate_url, data=data_output_text).json()["translations"][0]["text"]

        conversation_context = build_session_context(session)

        prompt = (
            f"Te rog sa analizezi urmatoarea pereche afirmatie/intrebare - raspuns. "
            f"Rezuma raspunsul si fa-l sa sune mai coerent. Reformuleaza raspunsul la persoana a 2-a ca si cum ai fi un terapeut care vorbeste cu un pacient. "
            f"Te rog sa imi returnezi doar raspunsul."
            f"Analizeaza profilul utilizatorului si adapteaza raspunsul doar unde este relevant. Arata-i ca estinteresat de preocuparile sale."
            f"Daca intrebarea este simpla sau factuala, raspunde concis si evita divagarea emotionala"
            f"Nu include repetitiv elemente din profil daca nu sunt legate direct de intrebarea pacientului.\n\n"
            f"Te rog sa analizezi profilul utilizatorului si sa adaptezi raspunsul la acesta:\n{user_profile_text}\n\n"
            f"Contextul conversatiei:\n{conversation_context}\n\n"
            f"Daca ti se pare ca o anumita parte din raspuns nu are nicio legatura cu contextul discutiei, elimina acea parte."
        )

        contents = [
            types.Content(role="user", parts=[types.Part(text=prompt)]),
            types.Content(role="user", parts=[types.Part(text=query)]),
            types.Content(role="user", parts=[types.Part(text=output_text)]),
        ]

        response = ""
        for chunk in client.models.generate_content_stream(
                model=gemini_model,
                contents=contents
        ):
            response += chunk.text

        final_response = response.strip()
        session.append([query, final_response])

        print("Terapeutul:", final_response)
        print("-" * 50)

if __name__ == "__main__":
    main()