from locust import HttpUser, task, between
import random
import json


class HateSpeechUser(HttpUser):
    wait_time = between(1, 2)

    @task
    def predict(self):
        texts = [
            "I love sunny days",
            "I hate when people are rude",
            "This is a neutral statement",
            "You are so annoying",
            "Have a nice day!",
            "I despise that behavior",
        ]
        text = random.choice(texts)
        payload = {"text": text}
        headers = {"Content-Type": "application/json"}

        response = self.client.post("/predict", data=json.dumps(payload), headers=headers)

        if response.status_code != 200:
            print(f"Error: Received status code {response.status_code}")
