#%% import required libraries
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

model_name = "deepset/roberta-base-squad2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)
res = nlp({'question': 'Where do I live?', 'context': 'My name is Wolfgang and I live in Berlin'})
print(res)
# #%% load model from local directory if it works
# model = TFAutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
# print("-----------  model loaded from local dir ------------")
# tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
# print("-----------  tokenizer loaded from local dir ------------")
# classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

# classifier(["good"]) 
