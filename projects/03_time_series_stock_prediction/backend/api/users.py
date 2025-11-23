"""User management API endpoints."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from ..core.database import get_db
from ..core.security import get_current_active_user, get_password_hash
from ..models.database import User

router = APIRouter()


class UserUpdate(BaseModel):
    full_name: str = None
    company: str = None


class PasswordChange(BaseModel):
    current_password: str
    new_password: str


@router.patch("/me")
async def update_user(
    user_data: UserUpdate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Update current user profile."""
    if user_data.full_name:
        current_user.full_name = user_data.full_name
    if user_data.company:
        current_user.company = user_data.company

    db.commit()
    db.refresh(current_user)
    return current_user


@router.post("/me/change-password")
async def change_password(
    password_data: PasswordChange,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Change user password."""
    from ..core.security import verify_password

    if not verify_password(password_data.current_password, current_user.hashed_password):
        raise HTTPException(status_code=400, detail="Incorrect password")

    current_user.hashed_password = get_password_hash(password_data.new_password)
    db.commit()

    return {"message": "Password changed successfully"}


@router.delete("/me")
async def delete_account(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Delete user account."""
    db.delete(current_user)
    db.commit()
    return {"message": "Account deleted successfully"}
