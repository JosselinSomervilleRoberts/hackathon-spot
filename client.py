from abc import abstractmethod, ABC


class Client(ABC):

    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.api_key = api_key

    @abstractmethod
    def make_request(self, prompt: str):
        pass
