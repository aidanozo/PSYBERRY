import xml.etree.ElementTree as ET
from pathlib import Path
import json
from datetime import datetime
from edit_profile.update import update_profile_from_session


def count_filled_profile_fields():
    profile_path = Path(__file__).parent / "edit_profile" / "profile.xml"
    if not profile_path.exists():
        return 0

    tree = ET.parse(profile_path)
    root = tree.getroot()

    count = 0
    for elem in root.iter():
        if elem.text and elem.text.strip():
            count += 1
    return count


def start_profile_initialization_session():
    print("Terapeutul: Buna! Bine ai venit in cabinetul meu. Sunt PSYBERRY, terapeutul tau si sunt gata sa te ajut. Spune-mi catvea lucuri despre tine pentru a ne putea cunoaste mai bine. :)")
    sub_session = []


    user_input = input("Tu: ").strip()


    response = "Multumesc ca mi-ai impartasit asta. Acum putem continua discutia noastra. Ce te framanta in ultima perioada?"
    sub_session.append([user_input, response])


    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"session_autofill_{timestamp}.json"

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(sub_session, f, ensure_ascii=False, indent=2)

    update_profile_from_session(session_folder=".")
    print(f"\n{response}")
