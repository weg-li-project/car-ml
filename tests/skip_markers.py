import os
import pytest

from util.paths import vision_api_results_path, charges_schroeder_path


needs_private_testdata = pytest.mark.skipif(
    not (
        os.path.exists(vision_api_results_path)
        and os.path.exists(charges_schroeder_path)
    ),
    reason="Charges or vision api results are missing",
)

needs_google_credentials = pytest.mark.skipif(
    not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"),
    reason="Google Application Credentials are missing",
)
