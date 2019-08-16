from typing import Any

import json

def load_input_file(input_file: str) -> Any:
    if input_file.endswith('json'):
        with open(input_file, 'r') as f:
            data = json.load(f)

    return data
