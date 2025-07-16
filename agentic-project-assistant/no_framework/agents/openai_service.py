from openai import OpenAI


class OpenAIService:
    """Simple wrapper around the OpenAI client used by the agents."""

    def __init__(self, api_key: str, base_url: str = "https://openai.vocareum.com/v1"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def chat(self, **kwargs):
        """Call the chat completion endpoint."""
        return self.client.chat.completions.create(**kwargs)

    def embed(self, **kwargs):
        """Call the embeddings endpoint."""
        return self.client.embeddings.create(**kwargs)
