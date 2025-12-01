from typing import Dict
from fastapi import APIRouter

router = APIRouter()


@router.get(path="/", response_model=dict[str, str])
def read_root() -> Dict:
    """
    Root URL greeting endpoint
    """
    return {"message": "Hello from FastAPI!"}
