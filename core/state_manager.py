import json
import os
from datetime import datetime

STATE_FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "nexus_state.json")

class StateManager:
    def __init__(self):
        self.state = {}
        self.load()

    def load(self):
        if os.path.exists(STATE_FILE_PATH):
            try:
                with open(STATE_FILE_PATH, 'r') as f:
                    self.state = json.load(f)
            except Exception as e:
                print(f"[STATE] ⚠️ Error loading persistent memory: {e}")
                self.state = {}
        else:
            self.state = {}

    def save(self):
        try:
            os.makedirs(os.path.dirname(STATE_FILE_PATH), exist_ok=True)
            with open(STATE_FILE_PATH, 'w') as f:
                json.dump(self.state, f, indent=4)
        except Exception as e:
            print(f"[STATE] ⚠️ Error saving persistent memory: {e}")

    def get(self, key, default=None):
        return self.state.get(key, default)

    def set(self, key, value):
        self.state[key] = value
        self.save()

# Global instance for easy import across modules
nexus_state = StateManager()
