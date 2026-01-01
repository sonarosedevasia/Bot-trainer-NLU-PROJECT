from pydantic import BaseModel, EmailStr
from typing import List, Optional

# User creation/input
class UserCreate(BaseModel):
    email: EmailStr
    password: str

# User output (exclude password)
class UserOut(BaseModel):
    id: int
    email: EmailStr

    class Config:
        orm_mode = True

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

class DatasetCreate(BaseModel):
    name: str
    file_path: str

class DatasetOut(BaseModel):
    id: int
    name: str
    file_path: str

    class Config:
        orm_mode = True

class BotCreate(BaseModel):
    name: str
    domain: Optional[str] = None

class BotOut(BaseModel):
    id: int
    name: str
    domain: Optional[str] = None
    datasets: List[DatasetOut] = []

    class Config:
        orm_mode = True
# ---------------------------
# Feedback Schemas
# ---------------------------

class FeedbackCreate(BaseModel):
    bot_id: int
    feedback_text: str


class FeedbackOut(BaseModel):
    id: int
    user_email: EmailStr
    bot_id: int
    feedback_text: str
    timestamp: str

    class Config:
        orm_mode = True
