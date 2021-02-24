import os
import pytest

from util.paths import vision_api_results_path, charges_schroeder_path


needs_private_testdata = pytest.mark.skipif(
    not (
        os.path.exists(vision_api_results_path)
        and os.path.exists(charges_schroeder_path)
    ),
    reason="Charges or vision api results are missing.",
)

needs_google_integration = pytest.mark.skipif(
    not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    or not os.getenv("INTEGRATION_TEST", "False").strip().lower() == "true",
    reason="Google Application Credentials are missing.",
)

needs_performance_flag = pytest.mark.skipif(
    not os.getenv("PERFORMANCE_TEST", "False").strip().lower() == "true",
    reason="PERFORMANCE_TEST environment variable not true.",
)
