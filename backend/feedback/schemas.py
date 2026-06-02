"""Schemas Pydantic para los endpoints de feedback."""

from typing import Optional, Union

from pydantic import BaseModel, Field

class FeedbackRequest(BaseModel):
    """Payload de POST /feedback.

    `value` admite bool (thumbs up/down), float (escala numerica) o str
    (preferencia A/B/empate del comparador). `metadata` permite enriquecer
    el assessment con campos arbitrarios (p.ej. trace_id_a/trace_id_b en
    el comparador).
    """

    trace_id: str = Field(..., min_length=1, description="request_id del trace MLflow al que se asocia el feedback.")
    value: Union[bool, float, str]
    source_id: str = Field(..., min_length=1, description="Identificador del usuario que emite la valoracion.")
    name: str = Field(default="user_thumbs", min_length=1)
    rationale: Optional[str] = None
    metadata: dict[str, str] = Field(default_factory=dict)

class FeedbackResponse(BaseModel):
    """Respuesta de POST /feedback."""

    status: str
    detail: Optional[str] = None

class ExpectationRequest(BaseModel):
    """Payload de POST /feedback/expectation.

    Permite anotar `expected_response` y/o `expected_facts` para un trace en
    una sola llamada. Al menos uno de los dos campos debe venir poblado.
    """

    trace_id: str = Field(..., min_length=1)
    expected_response: Optional[str] = None
    expected_facts: Optional[list[str]] = None
    source_id: str = Field(default="frontend_user", min_length=1)

class ExpectationResponse(BaseModel):
    """Respuesta de POST /feedback/expectation."""

    status: str
    registered: list[str] = Field(default_factory=list)
    detail: Optional[str] = None
