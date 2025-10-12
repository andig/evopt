from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class OptimizerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OPTIMIZER_")

    num_threads: int | None = Field(default=None, description="Number of threads to use for optimization")
    time_limit: float | None = Field(default=None, description="Time limit for the optimization process in seconds")
