import json

class Config:
    def __init__(self, config_path="conf.json"):
        self.config_path = config_path
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)
    
    def get(self, *keys, default=None):
        """Fetch nested keys safely, return default if not found."""
        data = self.config
        for key in keys:
            data = data.get(key, default) if isinstance(data, dict) else default
        return data

    def update(self, value, *keys):
        """Update nested key with a new value."""
        data = self.config
        for key in keys[:-1]:
            if key not in data or not isinstance(data[key], dict):
                data[key] = {}   # create nested dict if missing
            data = data[key]
        data[keys[-1]] = value
    
    def save(self):
        """Write updated config back to file."""
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=4)