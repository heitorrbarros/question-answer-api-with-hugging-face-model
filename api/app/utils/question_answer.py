from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from app.utils.logger import logger

model_name = 'deepset/roberta-base-squad2'

def predict_answer(question, context):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)
    res = nlp({'question': question, 'context': context})
    logger.debug(res)
    return res['answer']
