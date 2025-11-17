from locust import HttpUser, task, between
from pathlib import Path

class EfficientNetUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def predict(self):
        # Path to the image we want to send
        image_path = Path("locust_test_image.jpg")  

        # Opening the image in binary mode
        with image_path.open("rb") as f:
            files = {"file": (image_path.name, f, "image/jpeg")}
            # Sending POST request with multipart/form-data
            self.client.post("/predict", files=files)
