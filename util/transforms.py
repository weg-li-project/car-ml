import collections
import json
from typing import List, Dict


def to_json_suggestions(
    license_plate_numbers: List[str] = None,
    makes: List[str] = None,
    colors: List[str] = None,
) -> str:
    """Turns lists of license plate numbers and vehicle features into a JSON string."""
    return json.dumps(
        {
            "suggestions": {
                "license_plate_number": license_plate_numbers
                if license_plate_numbers
                else [],
                "make": makes if makes else [],
                "color": colors if colors else [],
            }
        }
    )


def order_by_frequency(seq: List[str]) -> List[str]:
    """Order a provided list of elements by frequency."""
    counts = collections.Counter(seq)
    return sorted(seq, key=counts.get, reverse=True)


def get_uniques(seq: List[str]) -> List[str]:
    """Return exclusively unique elements of a provided list and preserves the order."""
    uniques = set()
    return [x for x in seq if x not in uniques and (uniques.add(x) or True)]


def make_list(d: Dict[any, List[any]]) -> List[any]:
    """Turn lists from dictionary values into a single list."""
    return [entry for entries in d.values() for entry in entries]


def result_list(d: Dict[str, List[str]]) -> List[str]:
    """Return a list from the entries of the provided dictionary.
    Ordered by frequency and reduced to unique values only.
    """
    return get_uniques(order_by_frequency(make_list(d)))
