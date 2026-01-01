from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from fastapi.security import OAuth2PasswordRequestForm
from backend import schemas, crud, auth, database

router = APIRouter()

# Dependency to get DB session
def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ---------------------------
# Register New User
# ---------------------------
@router.post("/register", response_model=schemas.UserOut)
def register_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    existing_user = crud.get_user_by_email(db, user.email)
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    return crud.create_user(db, user)

# ---------------------------
# Login (works with OAuth2PasswordRequestForm)
# ---------------------------
@router.post("/login")
def login_user(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = crud.authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = auth.create_access_token({
        "sub": user.email,
        "role": user.role   # ⭐ ADD ROLE HERE
    })
    
    return {
        "access_token": token,
        "token_type": "bearer",
        "role": user.role,
    }

# ---------------------------
# Get Current User
# ---------------------------
@router.get("/me", response_model=schemas.UserOut)
def get_me(current_user=Depends(auth.get_current_user)):
    return current_user
