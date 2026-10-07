"""Pydantic models for POST /api/match, mirroring mock_api.json."""
from typing import Annotated, List, Literal

from pydantic import BaseModel, Field

Role = Literal['Business Analyst', 'Data Analyst', 'Data Architect', 'Data Engineer', 'Data Scientist', 'Machine Learning Engineer']
Level = Annotated[float, Field(ge=1, le=5, description='Self-rated level, 1 to 5')]
TipiAnswer = Annotated[int, Field(ge=1, le=7)]


class Skills(BaseModel):
    big_data: Level
    maths_stats: Level
    coding: Level
    ai_ml: Level
    dashboard_story: Level


class MatchRequest(BaseModel):
    name: str = 'You'
    target_role: Role
    hours_per_week: Annotated[int, Field(ge=1, le=40)]
    city: Annotated[str, Field(min_length=1, max_length=60)]
    skills: Skills
    tipi: Annotated[List[TipiAnswer], Field(min_length=10, max_length=10, description='10 TIPI answers, each 1 to 7')]


class GapItem(BaseModel):
    area: str
    label: str
    your_level: float
    role_weight: float
    weighted_gap: float


class UserOut(BaseModel):
    learner_persona: str
    workstyle_persona: str
    traits_sds_scale: dict[str, int]
    trait_success_prob: float
    solo_hike_prob: float
    gap: List[GapItem]


class Member(BaseModel):
    name: str
    is_user: bool
    skills: dict[str, float]
    hours_per_week: int
    strongest_area: str
    learner_persona: str
    workstyle_persona: str
    trait_success_prob: float
    accountability_lead: bool


class Parts(BaseModel):
    coverage: float
    overlap: float
    hours: float
    traits: float


class GroupOut(BaseModel):
    members: List[Member]
    score: float
    parts: Parts
    group_hike_prob: float
    explanation: List[str]
    uncovered_areas: List[str]
    accountability_note: str = 'Accountability lead rotates weekly'


class Week(BaseModel):
    week: int
    focus_area: str
    lead: str
    hours: int
    goals: List[str]


class Plan(BaseModel):
    weeks: List[Week]


class Project(BaseModel):
    title: str
    areas: List[str]
    gap_closed_pct: float
    why: str


class RoleCircle(BaseModel):
    role: str
    groups: int
    members: int


class CityCircle(BaseModel):
    city: str
    data_role_postings: int
    groups_in_city: int


class Community(BaseModel):
    role_circle: RoleCircle
    city_circle: CityCircle
    mentor_pool: int


class MatchResponse(BaseModel):
    user: UserOut
    group: GroupOut
    plan: Plan
    project: Project
    community: Community
