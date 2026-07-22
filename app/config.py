"""Application configuration, loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration.

    Values are read from environment variables or a local `.env` file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- OpenAI / generation ---
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    generation_model: str = Field(default="gpt-4o-mini", alias="GENERATION_MODEL")
    max_tokens: int = Field(default=1024, alias="MAX_TOKENS")

    # --- Embeddings (OpenAI) ---
    embedding_model: str = Field(
        default="text-embedding-3-small", alias="EMBEDDING_MODEL"
    )

    # --- Chroma vector store ---
    chroma_path: str = Field(default="./data/chroma", alias="CHROMA_PATH")
    collection_name: str = Field(default="documents", alias="COLLECTION_NAME")

    # --- Supabase (Pasha databank source of truth) ---
    # Used to pull fresh databank rows when syncing them into the vector store.
    supabase_url: str = Field(default="", alias="SUPABASE_URL")
    supabase_service_role_key: str = Field(
        default="", alias="SUPABASE_SERVICE_ROLE_KEY"
    )

    # --- Retrieval / guardrails ---
    top_k: int = Field(default=4, alias="TOP_K")
    # Chroma returns cosine *distance* (0 = identical, 2 = opposite). We refuse to
    # answer when the best match is farther than this threshold — i.e. nothing in
    # the corpus is relevant enough to ground an answer. The model's own grounding
    # check is the second line of defence.
    #
    # Calibrated against text-embedding-3-large on the pasha.org.pk corpus: real
    # questions land at 0.14-0.52, off-topic ones at 0.70+. This sits in that gap.
    # It is model-specific — re-measure if EMBEDDING_MODEL changes.
    max_distance: float = Field(default=0.60, alias="MAX_DISTANCE")

    # Simple API-key gate for the service itself (separate from Anthropic's key).
    service_api_key: str = Field(default="", alias="SERVICE_API_KEY")

    @property
    def auth_enabled(self) -> bool:
        return bool(self.service_api_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()
