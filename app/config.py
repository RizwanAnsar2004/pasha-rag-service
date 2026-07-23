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
    generation_model: str = Field(default="gpt-5.4-mini", alias="GENERATION_MODEL")
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
    # 8, not 4: the website corpus puts several near-duplicate chunks (taxonomy
    # archives, repeated press releases) above the one page that actually answers
    # a role question, so the right chunk lands around rank 6. Measured with
    # `scripts.eval_coverage` — 4 scores 91%, 8 scores 97%, with no loss of
    # refusals on the negative controls.
    top_k: int = Field(default=8, alias="TOP_K")
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

    # --- Rate limiting (/query) ---
    # Primary bucket: the caller's session_id, so one visitor's questions are
    # counted together and separately from everyone else's. Keyed on the IP only
    # when no session id was sent.
    query_rate_limit: str = Field(default="5/minute", alias="QUERY_RATE_LIMIT")
    # Backstop bucket: the client IP, which still holds when someone rotates
    # session ids to escape the limit above. Deliberately looser — several
    # people can share one NAT/office IP, so this must not be the binding
    # constraint for ordinary use.
    #
    # Two windows, because one can't do both jobs: 60/minute absorbs a burst
    # from a busy shared address (12x the per-session rate), while 500/hour caps
    # what a session-id-rotating caller can actually spend — without it, the
    # minute window alone would permit 3,600 generated answers an hour from one
    # address. Multiple limits are `;` separated.
    query_ip_rate_limit: str = Field(
        default="60/minute;500/hour", alias="QUERY_IP_RATE_LIMIT"
    )

    @property
    def auth_enabled(self) -> bool:
        return bool(self.service_api_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()
