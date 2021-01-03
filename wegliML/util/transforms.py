import collections
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


def order_by_frequency(seq: List[str]) -> List[str]:
    """Orders a provided list of elements by frequency."""
    counts = collections.Counter(seq)
    return sorted(seq, key=counts.get, reverse=True)


def get_uniques(seq: List[str]) -> List[str]:
    """Returns exclusively unique elements of a provided list and preserves the order."""
    uniques = set()
    return [x for x in seq if x not in uniques and (uniques.add(x) or True)]
