from pydantic import BaseModel, field_validator


class GenerateRequest(BaseModel):
    video_id: str

    @field_validator('video_id')
    @classmethod
    def video_id_must_be_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError('video_id must be a non-empty string')
        return v.strip()


class Timestamp(BaseModel):
    time: str   # "M:SS" or "H:MM:SS"
    title: str


class GenerateResponse(BaseModel):
    timestamps: list[Timestamp]
