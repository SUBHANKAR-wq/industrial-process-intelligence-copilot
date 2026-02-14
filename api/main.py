from fastapi import FastAPI
from api.routes import router

app = FastAPI(
    title="Industrial Process Intelligence API"
)

app.include_router(router)