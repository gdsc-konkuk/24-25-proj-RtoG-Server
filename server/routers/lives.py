from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from database import get_db
from services import get_lives

router = APIRouter()

@router.get("/lives")
async def lives_endpoint(db: Session = Depends(get_db)):
    return get_lives(db) 