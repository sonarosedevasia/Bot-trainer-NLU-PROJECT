from sqlalchemy import Column, Integer, String, ForeignKey, Text
from sqlalchemy.orm import relationship
from .database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    password = Column(String, nullable=False)
    role = Column(String, default="user")   # NEW COLUMN

    bots = relationship("Bot", back_populates="owner")


class Bot(Base):
    __tablename__ = "bots"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True, nullable=False)
    domain = Column(String, nullable=True)
    owner_id = Column(Integer, ForeignKey("users.id"))

    owner = relationship("User", back_populates="bots")
    datasets = relationship("Dataset", back_populates="bot")
    annotations = relationship("Annotation", back_populates="bot", cascade="all, delete")


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    bot_id = Column(Integer, ForeignKey("bots.id"))

    bot = relationship("Bot", back_populates="datasets")


class Annotation(Base):
    __tablename__ = "annotations"

    id = Column(Integer, primary_key=True, index=True)
    bot_id = Column(Integer, ForeignKey("bots.id"))
    text = Column(String, nullable=False)
    intent = Column(String, nullable=False)
    entities = Column(Text, nullable=True)
    manual_entities = Column(Text, nullable=True)

    bot = relationship("Bot", back_populates="annotations")
class ActivityLog(Base):
    __tablename__ = "activity_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_email = Column(String, nullable=False)
    action = Column(Text, nullable=False)
    timestamp = Column(String, nullable=False)   # simple string timestamp

class Feedback(Base):
    __tablename__ = "feedbacks"

    id = Column(Integer, primary_key=True, index=True)
    user_email = Column(String, nullable=False)
    bot_id = Column(Integer, ForeignKey("bots.id"), nullable=False)
    feedback_text = Column(Text, nullable=False)
    timestamp = Column(String, nullable=False)
