import json
from typing import List, Final

from flask import abort, Request

PROP_NAME: Final = 'google_cloud_urls'


def alpr_analysis(request: Request):
    """Responds to POST HTTP request containing a JSON body with Google Cloud Storage URLs.
    Args:
        request (flask.Request): HTTP request object.
    Returns:
        A JSON string that contains suggestions for the license plate and vehicle features.
    """
    if request.method != 'POST':
        return abort(400)

    content_type = request.headers['content-type']
    if content_type != 'application/json':
        return abort(400)

    request_json = request.get_json(silent=True)
    if not request_json or PROP_NAME not in request_json:
        raise ValueError(f"JSON is invalid, or missing a '{PROP_NAME}' property")
    google_cloud_urls: List[str] = request_json[PROP_NAME]

    if len(google_cloud_urls) < 1:
        abort(400)

    return alpr_magic(google_cloud_urls=google_cloud_urls)


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


def alpr_magic(google_cloud_urls: List[str]) -> str:
    # Your python voodoo magic to get license plate
    print(google_cloud_urls)

    return to_json_suggestions()
