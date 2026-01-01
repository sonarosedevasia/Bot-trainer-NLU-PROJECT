from sqlalchemy.orm import Session
from backend import models, schemas
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# -------- User --------
def get_user_by_email(db: Session, email: str):
    return db.query(models.User).filter(models.User.email == email).first()


def create_user(db: Session, user: schemas.UserCreate):
    hashed_password = pwd_context.hash(user.password)
    db_user = models.User(
        email=user.email,
        password=hashed_password,
        role="user"   # NEW DEFAULT
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


def authenticate_user(db: Session, email: str, password: str):
    user = get_user_by_email(db, email)
    if not user or not pwd_context.verify(password, user.password):
        return None
    return user


# -------- Bot --------
def create_bot(db: Session, owner_id: int, bot: schemas.BotCreate):
    db_bot = models.Bot(name=bot.name, domain=bot.domain, owner_id=owner_id)
    db.add(db_bot)
    db.commit()
    db.refresh(db_bot)
    return db_bot


def get_bots(db: Session, owner_id: int):
    return db.query(models.Bot).filter(models.Bot.owner_id == owner_id).all()


def delete_bot(db: Session, bot_id: int, owner_id: int):
    bot = db.query(models.Bot).filter(models.Bot.id == bot_id, models.Bot.owner_id == owner_id).first()
    if bot:
        db.delete(bot)
        db.commit()
        return True
    return False


# -------- Dataset --------
def add_dataset(db: Session, bot_id: int, dataset: schemas.DatasetCreate):
    db_dataset = models.Dataset(name=dataset.name, file_path=dataset.file_path, bot_id=bot_id)
    db.add(db_dataset)
    db.commit()
    db.refresh(db_dataset)
    return db_dataset


def get_datasets(db: Session, bot_id: int):
    return db.query(models.Dataset).filter(models.Dataset.bot_id == bot_id).all()


def get_dataset_by_id(db: Session, dataset_id: int):
    return db.query(models.Dataset).filter(models.Dataset.id == dataset_id).first()


def get_all_datasets_by_user(db: Session, user_id: int):
    return db.query(models.Dataset).join(models.Bot).filter(models.Bot.owner_id == user_id).all()
