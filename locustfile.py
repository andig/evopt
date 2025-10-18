import json
import random
from pathlib import Path

from locust import HttpUser, task

scenarios = [json.loads(path.read_text())["request"] for path in Path("test_cases").glob("*.json")]


class EVCCUser(HttpUser):
    @task
    def optimize_charge_schedule(self):
        scenario = random.choice(scenarios)
        self.client.post("/optimize/charge-schedule", json=scenario)
