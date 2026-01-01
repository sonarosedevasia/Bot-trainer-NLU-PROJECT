from datetime import datetime, timedelta
from jose import jwt, JWTError
from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from . import crud, database

# ---------------------------------------------
# JWT CONFIG
# ---------------------------------------------
SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")


# ---------------------------------------------
# DB Dependency
# ---------------------------------------------
def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------------------------------------------
# TOKEN CREATION
# ---------------------------------------------
def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


# ---------------------------------------------
# TOKEN DECODING
# ---------------------------------------------
def decode_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None


# ---------------------------------------------
# GET CURRENT USER (email + role)
# ---------------------------------------------
def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    """
    Extract BOTH email and role from the JWT token.
    Attach role to the user object so admin checks work.
    """

    payload = decode_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid token")

    email: str = payload.get("sub")
    role: str = payload.get("role", "user")

    if email is None:
        raise HTTPException(status_code=401, detail="Invalid token")

    user = crud.get_user_by_email(db, email)
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")

    # 🔥 Attach token role to user instance
    user.role = role

    return user
