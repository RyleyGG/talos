from pathlib import Path

from pydantic_settings import BaseSettings


__root__ = Path(__file__).parent.parent


class AppSettings(BaseSettings):
    vosk_models_path: Path = __root__ / '.vosk_models'
settings = AppSettings()
