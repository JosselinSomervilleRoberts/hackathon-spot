from typing import Dict
from client import Client

import requests


class TogetherClientError(Exception):
    pass


class TogetherClient(Client):
    """
    Client for the models where we evaluate offline. Since the queries are handled offline, the `TogetherClient` just
    checks if the request/result is cached. We return the result if it's in the cache. Otherwise, we return an error.
    """

    INFERENCE_ENDPOINT: str = "https://api.together.xyz/api/inference"

    def __init__(self, model_name: str, api_key: str):
        self.api_key: str = api_key
        self.model_name = model_name

    def _get_job_url(self, job_id: str) -> str:
        return f"https://api.together.xyz/jobs/job/{job_id}"

    def make_request(self, prompt: str):
        raw_request = {
            "request_type": "language-model-inference",
            "model": self.model_name,
            "prompt": prompt,
        }

        if not self.api_key:
            raise TogetherClientError("togetherApiKey not set in credentials.conf")
        headers: Dict[str, str] = {"Authorization": f"Bearer {self.api_key}"}

        try:
            response = requests.post(
                TogetherClient.INFERENCE_ENDPOINT, headers=headers, json=raw_request
            )
            try:
                response.raise_for_status()
            except Exception as e:
                raise TogetherClientError(
                    f"Together request failed with {response.status_code}: {response.text}"
                ) from e
            result = response.json()
            if "output" not in result:
                raise TogetherClientError(
                    f"Could not get output from Together response: {result}"
                )
            if "error" in result["output"]:
                error_message = result["output"]["error"]
                raise TogetherClientError(
                    f"Together request failed with error: {error_message}"
                )
            return result["output"]["choices"][0]["text"], None

        except Exception as error:
            return "", error
