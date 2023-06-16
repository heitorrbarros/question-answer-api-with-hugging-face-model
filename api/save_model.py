#%% import required libraries
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

model_name = "deepset/roberta-base-squad2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)
res = nlp({'question': 'Where do I live?', 'context': 'My name is Wolfgang and I live in Berlin'})
print(res)
