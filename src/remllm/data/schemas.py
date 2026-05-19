"""Data schemas (Pydantic models) for training and evaluation records."""

from pydantic import BaseModel, Field


class TrainingRow(BaseModel):
    instruction: str
    input: str = ""
    output: str


class EvalRow(BaseModel):
    instruction: str
    input: str = ""
    output: str = ""


class MultiFileContext(BaseModel):
    existing_files: dict[str, str] = Field(default_factory=dict)
    constraints: list[str] = Field(default_factory=list)


class MultiFileOutputFile(BaseModel):
    path: str
    content: str
    action: str = "create"


class MultiFileOutput(BaseModel):
    files: dict[str, str] = Field(default_factory=dict)
    explanation: str = ""


class WebTrainingRow(BaseModel):
    instruction: str
    context: MultiFileContext = Field(default_factory=MultiFileContext)
    output: MultiFileOutput = Field(default_factory=MultiFileOutput)
    domain: str = "general"
    difficulty: str = "easy"
    tags: list[str] = Field(default_factory=list)
