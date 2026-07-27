import json
import os

def save_project_state(state_dict: dict, file_path: str = "data/processed/project_state.json"):
    """Saves the essential Streamlit session state to disk for persistence."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(state_dict, f, indent=4)
    except Exception as e:
        print(f"Failed to save project state: {e}")

def load_project_state(file_path: str = "data/processed/project_state.json") -> dict:
    """Loads the Streamlit session state from disk if it exists."""
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Failed to load project state: {e}")
    return {}
