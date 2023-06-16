from fastapi import FastAPI
from app.schema.question import Question, QuestionAnswer
from app.utils.question_answer import predict_answer

app = FastAPI()


@app.post("/answer")
async def answer(body : Question) -> QuestionAnswer:
    answer = predict_answer(body.question, body.context)
    return QuestionAnswer(answer=answer)