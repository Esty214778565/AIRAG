"""
Pydantic schema for the structured data layer (Stage C - Data Extraction).

Two families of models live here:

- Extracted*  - the shape the LLM fills in for a single markdown section.
  Deliberately free of ids/sources/timestamps: the LLM should only ever
  describe *what* it found, never invent bookkeeping metadata.
- *Item       - the persisted record, made of an Extracted* payload plus
  metadata assigned deterministically in Python (id, source, observed_at).

Four item types are supported: decisions, rules, warnings, dependencies.
"""

from typing import List

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1.0"
ITEM_TYPES = ("decisions", "rules", "warnings", "dependencies")


# --- LLM-facing extraction models ------------------------------------------

class ExtractedDecision(BaseModel):
    title: str = Field(description="Short title of the technical decision")
    summary: str = Field(description="1-2 sentence summary of what was decided and why")
    tags: List[str] = Field(default_factory=list, description="Topic tags, e.g. ['db', 'architecture']")


class ExtractedRule(BaseModel):
    rule: str = Field(description="The rule/guideline itself, as a short imperative statement")
    scope: str = Field(description="Area the rule applies to, e.g. 'ui', 'naming', 'testing'")
    notes: str = Field(default="", description="Exceptions or extra context, empty string if none")


class ExtractedWarning(BaseModel):
    area: str = Field(description="Area the warning applies to, e.g. 'auth', 'payments'")
    message: str = Field(description="The warning / sensitivity note itself")
    severity: str = Field(description="One of: low, medium, high")


class ExtractedDependency(BaseModel):
    name: str = Field(description="Name of the dependency/technology/service")
    purpose: str = Field(description="What it is used for in this project")
    category: str = Field(description="Free-form category, e.g. 'runtime', 'aws', 'frontend', 'backend'")


class SectionExtraction(BaseModel):
    """Everything the LLM found in a single markdown section."""
    decisions: List[ExtractedDecision] = Field(default_factory=list)
    rules: List[ExtractedRule] = Field(default_factory=list)
    warnings: List[ExtractedWarning] = Field(default_factory=list)
    dependencies: List[ExtractedDependency] = Field(default_factory=list)


# --- Persisted store models --------------------------------------------------

class SourceRef(BaseModel):
    tool: str
    file: str
    anchor: str
    line_range: List[int]  # [start, end], 1-indexed inclusive


class DecisionItem(ExtractedDecision):
    id: str
    source: SourceRef
    observed_at: str


class RuleItem(ExtractedRule):
    id: str
    source: SourceRef
    observed_at: str


class WarningItem(ExtractedWarning):
    id: str
    source: SourceRef
    observed_at: str


class DependencyItem(ExtractedDependency):
    id: str
    source: SourceRef
    observed_at: str


class Items(BaseModel):
    decisions: List[DecisionItem] = Field(default_factory=list)
    rules: List[RuleItem] = Field(default_factory=list)
    warnings: List[WarningItem] = Field(default_factory=list)
    dependencies: List[DependencyItem] = Field(default_factory=list)


class FileMeta(BaseModel):
    path: str
    last_modified: str
    hash: str


class ToolSource(BaseModel):
    tool: str
    root_path: str
    files: List[FileMeta] = Field(default_factory=list)


class StructuredStore(BaseModel):
    schema_version: str = SCHEMA_VERSION
    generated_at: str
    sources: List[ToolSource] = Field(default_factory=list)
    items: Items = Field(default_factory=Items)
