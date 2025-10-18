
import json
import pathlib

import numpy
import pytest

from evopt.app import app


@pytest.mark.parametrize('test_case', pathlib.Path('test_cases').glob('*.json'))
def test_optimizer(test_case: pathlib.Path):
    client = app.test_client()

    test_data = json.loads(test_case.read_text())

    request = test_data["request"]
    expected_response = test_data.get("expected_response")

    response = client.post("/optimize/charge-schedule", json=request)

    assert response.status_code == 200

    if expected_response is not None:
        assert response.json["status"] == expected_response.get("status", {})
        assert numpy.isclose(response.json["objective_value"],
                             expected_response.get("objective_value", {}),
                             rtol=1e-05, atol=1e-08, equal_nan=False)
