# =========================
# Pydantic Models
# Used for request/response validation
# =========================

from pydantic import BaseModel
from typing import List


# Request model (input to /predict)
class PredictionRequest(BaseModel):
    recency: float
    frequency: float
    monetary: float

# Response model (output from API)
class PredictionResponse(BaseModel):
    risk_probability: float   # probability of default/risk
    risk_label: int           # 0 = low risk, 1 = high risk