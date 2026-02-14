from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router

app = FastAPI(
    title="Industrial Process Intelligence API"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # later you can restrict to React URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routes
app.include_router(router)


# health check endpoint (very useful)
@app.get("/")
def root():
    return {"status": "API running"}