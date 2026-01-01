from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

SQLALCHEMY_DATABASE_URL = "sqlite:///./chatbot.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# -------------------------------------------------------------------------------------
# REQUIRED BY ALL ROUTERS → This fixes your error
# -------------------------------------------------------------------------------------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# -------------------------------------------------------------------------------------
# Used in main.py at startup
# -------------------------------------------------------------------------------------
def init_db():
    from . import models  # ensure models are imported
    Base.metadata.create_all(bind=engine)
