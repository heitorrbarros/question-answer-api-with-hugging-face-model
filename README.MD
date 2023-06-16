# Requirements
* Docker
* Docker Compose

# Build
```sh
docker compose build . 
```

# Accessing API
After the **build** step you can use the command `docker compose up` to start the application and visit 
the api exposed at port 8000.

You can use `/docs` endpoint as the playground.

# Demo Questions
```json
[
  {
    "question": "What's the of company that Heitor works?",
    "context": "My name is Heitor and I live in São José dos Campos, São Paulo - Brazil. I'm a software engineering and I work at a company called 'Quero Educação'"
  }
]
```