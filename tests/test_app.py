
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

    assert response.status_code == 200, f"request returned with status {response.status_code}"

    if expected_response is not None:
        # check optimizer status
        actual_status = response.json["status"]
        expected_status = expected_response.get("status", {})
        assert actual_status == expected_status, \
            f"optimizer status: {actual_status}, expected was: {expected_status}"
        # check objective value
        actual_objective_value = response.json["objective_value"]
        expected_objective_value = expected_response.get("objective_value", {})
        assert numpy.isclose(actual_objective_value,
                             expected_objective_value,
                             rtol=1e-05, atol=1e-08, equal_nan=False), \
            f"objective value: {actual_objective_value}, expected was: {expected_objective_value}"
