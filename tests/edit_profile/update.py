import xml.etree.ElementTree as ET
import json
import os
from pathlib import Path
from google import genai
from google.genai import types


# Dupa ce termin de testat refac semnatura functiei:
def update_profile_from_session(session_folder: str = "."):

#def update_profile_from_session():

    profile_path = Path(__file__).parent / "profile.xml"

    gemini_key = os.environ["PSYBERRY_GEMINI_API_KEY"]
    client = genai.Client(api_key=gemini_key)
    gemini_model = "models/gemini-1.5-pro-latest"

    session_files = sorted(Path(session_folder).glob("session_*.json"), reverse=True)
    if not session_files:
        print("[WARN] Nicio sesiune gasita.")
        return

    latest_session_path = session_files[0]

    with open(latest_session_path, "r", encoding="utf-8") as f:
        session = json.load(f)

    session_text = ""
    for entry in session:
        user, therapist = entry
        session_text += f"Pacient: {user}\nTerapeut: {therapist}\n"

    with open(profile_path, "r", encoding="utf-8") as f:
        xml_structure = f.read()

    # Prompt catre Gemini
    prompt = (
        "Esti un psiholog digital. Ti se ofera o conversatie terapeutica si structura unui fisier XML. "
        "Returneaza un JSON unde cheile sunt XPath-uri absolute catre campuri din XML (ex: /user_profile/general_attributes/first_name) "
        "si valorile sunt completari relevante. Returneaza DOAR JSON-ul, fara explicatii.\n\n"
        "Structura XML:\n"
        f"{xml_structure}\n\n"
        "Conversatia:\n"
        f"{session_text}"
    )

    contents = [types.Content(role="user", parts=[types.Part(text=prompt)])]
    response = ""
    for chunk in client.models.generate_content_stream(model=gemini_model, contents=contents):
        response += chunk.text


    result = {}
    for line in response.splitlines():
        if ":" in line:
            parts = line.split(":", 1)
            key = parts[0].strip().strip('"')
            value = parts[1].strip().strip('",')
            if value.lower() != "null" and value != "":
                last_field = key.split("/")[-1]
                result[last_field] = value

    print(result)


    tree = ET.parse(profile_path)
    root = tree.getroot()


    for tag_name, value in result.items():

        elem = root.find(f".//{tag_name}")
        if elem is not None:
            elem.text = value
        else:
            print(f"[WARN] Nu am gasit tagul: {tag_name}")

    tree.write(profile_path, encoding="utf-8", xml_declaration=True)