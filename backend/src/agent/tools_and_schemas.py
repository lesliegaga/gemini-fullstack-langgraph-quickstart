from typing import List
from pydantic import BaseModel, Field


class SearchQueryList(BaseModel):
    query: List[str] = Field(
        description="A list of search queries to be used for web research."
    )
    rationale: str = Field(
        description="A brief explanation of why these queries are relevant to the research topic."
    )


class DualSearchQueryList(BaseModel):
    web_queries: List[str] = Field(
        description="A list of detailed search queries optimized for web search engines."
    )
    map_queries: List[str] = Field(
        description="A list of simplified location-based queries optimized for map APIs."
    )
    web_rationale: str = Field(
        description="A brief explanation of why the web queries are relevant to the research topic."
    )
    map_rationale: str = Field(
        description="A brief explanation of why the map queries are relevant for location-based search."
    )


class Reflection(BaseModel):
    is_sufficient: bool = Field(
        description="Whether the provided summaries are sufficient to answer the user's question."
    )
    knowledge_gap: str = Field(
        description="A description of what information is missing or needs clarification."
    )
    follow_up_queries: List[str] = Field(
        description="A list of follow-up queries to address the knowledge gap."
    )


class DualReflection(BaseModel):
    is_sufficient: bool = Field(
        description="Whether the provided summaries are sufficient to answer the user's question."
    )
    knowledge_gap: str = Field(
        description="A description of what information is missing or needs clarification."
    )
    web_follow_up_queries: List[str] = Field(
        description="A list of detailed follow-up queries for web search to address the knowledge gap."
    )
    map_follow_up_queries: List[str] = Field(
        description="A list of simplified location-based follow-up queries for map search."
    )
