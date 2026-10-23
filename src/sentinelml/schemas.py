from pydantic import BaseModel, ConfigDict, Field, field_validator
from sentinelml.config import FEATURE_NAMES

class Instance(BaseModel):
    model_config = ConfigDict(extra="forbid")
    feature_0: float
    feature_1: float
    feature_2: float
    feature_3: float
    feature_4: float
    feature_5: float
    feature_6: float
    feature_7: float
    feature_8: float
    feature_9: float

    @field_validator("*")
    @classmethod
    def finite_numbers(cls, value: float) -> float:
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("Feature values must be finite numbers.")
        return value

    def ordered_values(self) -> list[float]:
        return [getattr(self, name) for name in FEATURE_NAMES]

class PredictionRequest(BaseModel):
    instances: list[Instance] = Field(min_length=1, max_length=500)

class DriftRequest(BaseModel):
    instances: list[Instance] = Field(min_length=1, max_length=500)
