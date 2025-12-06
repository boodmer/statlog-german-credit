from typing import List, Literal
from pydantic import BaseModel, ConfigDict

GOOD_EXAMPLE = {
    "status": "A11",
    "duration": 12,
    "credit_history": "A34",
    "purpose": "A43",
    "credit_amount": 2100,
    "savings": "A65",
    "employment": "A75",
    "installment_rate": 2,
    "personal_status_sex": "A93",
    "other_debtors": "A101",
    "residence_since": 3,
    "property": "A121",
    "age": 35,
    "other_installment_plans": "A143",
    "housing": "A151",
    "existing_credits": 1,
    "job": "A173",
    "liable_maintenance_people": 1,
    "telephone": "A191",
    "foreign_worker": "A201"
}

BAD_EXAMPLE = {
    "status": "A14",
    "duration": 48,
    "credit_history": "A30",
    "purpose": "A40",
    "credit_amount": 9000,
    "savings": "A61",
    "employment": "A71",
    "installment_rate": 4,
    "personal_status_sex": "A92",
    "other_debtors": "A103",
    "residence_since": 1,
    "property": "A124",
    "age": 22,
    "other_installment_plans": "A142",
    "housing": "A153",
    "existing_credits": 3,
    "job": "A172",
    "liable_maintenance_people": 2,
    "telephone": "A192",
    "foreign_worker": "A202"
}


class Applicant(BaseModel):
    # Single default example (GOOD) for schema docs
    model_config = ConfigDict(
        json_schema_extra={
            "example": GOOD_EXAMPLE
        }
    )

    status: str
    duration: int
    credit_history: str
    purpose: str
    credit_amount: int
    savings: str
    employment: str
    installment_rate: int
    personal_status_sex: str
    other_debtors: str
    residence_since: int
    property: str
    age: int
    other_installment_plans: str
    housing: str
    existing_credits: int
    job: str
    liable_maintenance_people: int
    telephone: str
    foreign_worker: str


class PredictionResponse(BaseModel):
    predicted_label: Literal[0, 1]   # 1 = good, 0 = bad
    proba_good: float
    threshold: float


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
