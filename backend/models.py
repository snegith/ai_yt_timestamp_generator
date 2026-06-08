import re

from pydantic import BaseModel, field_validator

# Standard YouTube video IDs are exactly 11 URL-safe characters.
_YOUTUBE_VIDEO_ID_PATTERN = r"^[A-Za-z0-9_-]{11}$"


class GenerateRequest(BaseModel):
    video_id: str

    @field_validator('video_id')
    @classmethod
    def video_id_must_be_valid(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError('video_id must be a non-empty string')
        v = v.strip()
        if not re.fullmatch(_YOUTUBE_VIDEO_ID_PATTERN, v):
            raise ValueError(
                'video_id must be a valid 11-character YouTube video ID'
            )
        return v


class Timestamp(BaseModel):
    time: str   # "M:SS" or "H:MM:SS"
    title: str


class GenerateResponse(BaseModel):
    timestamps: list[Timestamp]
