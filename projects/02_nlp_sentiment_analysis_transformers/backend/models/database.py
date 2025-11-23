"""
Database models using SQLAlchemy.
Defines all database tables and relationships.
"""

from sqlalchemy import Column, Integer, String, Boolean, DateTime, Float, Text, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

Base = declarative_base()


class User(Base):
    """User model."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    uuid = Column(String, unique=True, index=True, default=lambda: str(uuid.uuid4()))
    email = Column(String, unique=True, index=True, nullable=False)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String)
    company = Column(String)

    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    is_verified = Column(Boolean, default=False)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = Column(DateTime)

    # Relationships
    api_keys = relationship("APIKey", back_populates="user", cascade="all, delete-orphan")
    predictions = relationship("Prediction", back_populates="user", cascade="all, delete-orphan")
    usage_stats = relationship("UsageStats", back_populates="user", cascade="all, delete-orphan")


class APIKey(Base):
    """API Key model for authentication."""
    __tablename__ = "api_keys"

    id = Column(Integer, primary_key=True, index=True)
    key = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    is_active = Column(Boolean, default=True)
    rate_limit_per_minute = Column(Integer, default=60)
    rate_limit_per_hour = Column(Integer, default=1000)

    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)
    last_used_at = Column(DateTime)

    # Relationships
    user = relationship("User", back_populates="api_keys")


class Prediction(Base):
    """Prediction history model."""
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    uuid = Column(String, unique=True, index=True, default=lambda: str(uuid.uuid4()))

    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    text = Column(Text, nullable=False)
    sentiment = Column(String, nullable=False)  # positive, negative, neutral
    confidence = Column(Float, nullable=False)
    probabilities = Column(JSON)

    model_version = Column(String)
    processing_time_ms = Column(Float)

    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    # Relationships
    user = relationship("User", back_populates="predictions")


class UsageStats(Base):
    """Usage statistics for analytics."""
    __tablename__ = "usage_stats"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    date = Column(DateTime, default=datetime.utcnow, index=True)
    endpoint = Column(String)

    total_requests = Column(Integer, default=0)
    successful_requests = Column(Integer, default=0)
    failed_requests = Column(Integer, default=0)

    avg_processing_time_ms = Column(Float)
    total_characters_processed = Column(Integer, default=0)

    # Relationships
    user = relationship("User", back_populates="usage_stats")


class Feedback(Base):
    """User feedback on predictions."""
    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(Integer, ForeignKey("predictions.id"))

    user_id = Column(Integer, ForeignKey("users.id"))
    correct_sentiment = Column(String)
    is_correct = Column(Boolean)
    comments = Column(Text)

    created_at = Column(DateTime, default=datetime.utcnow)


class Webhook(Base):
    """Webhook configuration."""
    __tablename__ = "webhooks"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    url = Column(String, nullable=False)
    name = Column(String)
    secret = Column(String)

    is_active = Column(Boolean, default=True)
    events = Column(JSON)  # ['prediction.created', 'prediction.batch.completed']

    created_at = Column(DateTime, default=datetime.utcnow)
    last_triggered_at = Column(DateTime)
