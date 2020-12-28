import json
from typing import List


def to_json_suggestions(license_plate_numbers: List[str] = None, makes: List[str] = None, colors: List[str] = None,
                        models: List[str] = None) -> str:
    """Turns lists of license plate numbers and vehicle features into a JSON string.
    """
    return json.dumps({
        'suggestions': {
            'license_plate_number': license_plate_numbers if license_plate_numbers else [],
            'make': makes if makes else [],
            'color': colors if colors else [],
            'model': models if models else []
        }
    }, indent=2)
