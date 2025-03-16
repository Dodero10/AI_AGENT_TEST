from typing import List, Field
from pydantic import BaseModel

class Reflection(BaseModel):
    missing: str = Field(description="Critique of what is missing in the text")
    superfluous: str = Field(description="Critique of what is superfluous in the text")  # Thừa thãi

class AnswerQuestion(BaseModel):
    """Answer the question based on the context"""

    answer: str = Field(description="Answer the question based on the context")
    reflection: Reflection = Field(description="Reflection on the answer")
    search_query: List[str] = Field(
        description="1- 3 search queries for researching improvements to address the critique of your current answer"
    )

    
