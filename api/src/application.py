from fastapi import FastAPI
from api.src.hello_svc.views import router

app = FastAPI(
    title="AI Stock Price Prediction",
    description="Project built with FastAPI framework.",
    version="0.1.0"
)

app.include_router(router)
