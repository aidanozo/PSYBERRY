import xml.etree.ElementTree as ET
from pathlib import Path

def extract_user_profile():
    profile_path = Path(__file__).parent / "profile.xml"
    if not profile_path.exists():
        return "[Profil indisponibil]"

    tree = ET.parse(profile_path)
    root = tree.getroot()

    profile_info = []

    for elem in root.iter():
        if elem.text and elem.text.strip():
            tag_path = elem.tag.replace("_", " ").capitalize()
            profile_info.append(f"{tag_path}: {elem.text.strip()}")

    return "\n".join(profile_info)