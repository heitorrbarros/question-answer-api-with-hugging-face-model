from pydantic import BaseModel

class Question(BaseModel):
    question: str
    context: str

class QuestionAnswer(BaseModel):
    answer: str