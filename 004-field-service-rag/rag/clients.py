"""OpenAI / Azure OpenAI client factory.

One place decides which SDK class to instantiate so nothing else in the codebase
has to know whether we are pointed at Azure or the public API.
"""
import os
from functools import lru_cache

from openai import AzureOpenAI, OpenAI

from rag.config import settings


@lru_cache(maxsize=1)
def get_client() -> OpenAI | AzureOpenAI:
    if settings.use_azure:
        return AzureOpenAI(
            azure_endpoint=settings.azure_endpoint,
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=settings.azure_api_version,
        )
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def embed(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts. Azure deployments are addressed by deployment name,
    which we keep identical to the model name to avoid a second config knob."""
    if not texts:
        return []
    client = get_client()
    response = client.embeddings.create(model=settings.embedding_model, input=texts)
    return [item.embedding for item in response.data]
