import os
print(">>> FASTAPI RUNNING FROM >", os.getcwd())

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from . import models, database
from .routers import admin_stats
from .routers import admin
from .routers import admin_logs
from .routers import feedback


from .routers import (
    users,
    workspace,
    train,
    evaluate,
    model_compare,
    active_learning,
)

models.Base.metadata.create_all(bind=database.engine)

app = FastAPI(title="Chatbot NLU Trainer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(users)
app.include_router(workspace)
app.include_router(train)
app.include_router(evaluate)
app.include_router(model_compare)
app.include_router(active_learning)
app.include_router(admin_stats.router)
app.include_router(admin.router)
app.include_router(admin_logs.router)
app.include_router(feedback.router)


@app.get("/")
def root():
    return {"message": "Backend working properly!"}
