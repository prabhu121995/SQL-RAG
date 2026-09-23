from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    groq_api_key: str
    groq_model: str = "openai/gpt-oss-20b"
    tickets_url: str = (
        "https://raw.githubusercontent.com/prabhu121995/sample-docs/"
        "refs/heads/main/tickets.db"
    )
    tickets_db_path: str = "./tickets.db"


settings = Settings()