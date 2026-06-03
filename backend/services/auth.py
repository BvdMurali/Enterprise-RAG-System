import logging
from datetime import datetime, timedelta
from typing import List, Optional
from jose import jwt, JWTError
from pydantic import BaseModel
from backend.config import get_settings

logger = logging.getLogger(__name__)

# Standalone mock user database for JWT authentication validation
USERS_DB = {
    "admin": {
        "username": "admin",
        "password_hash": "adminpass",  # Plain text checks for local dev purposes
        "user_groups": ["admin", "hr", "finance", "public"]
    },
    "hr_user": {
        "username": "hr_user",
        "password_hash": "hrpass",
        "user_groups": ["hr", "public"]
    },
    "finance_user": {
        "username": "finance_user",
        "password_hash": "financepass",
        "user_groups": ["finance", "public"]
    }
}


class TokenPayload(BaseModel):
    sub: str
    user_groups: List[str]
    exp: float


def create_access_token(username: str, expires_delta: Optional[timedelta] = None) -> str:
    """Generate a signed JWT token containing user scopes."""
    settings = get_settings()
    user = USERS_DB.get(username)
    if not user:
        raise ValueError("User not found")
        
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(hours=8)
        
    to_encode = {
        "sub": username,
        "user_groups": user["user_groups"],
        "exp": expire.timestamp()
    }
    encoded_jwt = jwt.encode(to_encode, settings.jwt_secret, algorithm=settings.jwt_algorithm)
    return encoded_jwt


def verify_token(token: str) -> Optional[TokenPayload]:
    """Decode and validate the signature and expiry of a JWT token."""
    settings = get_settings()
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
        username: str = payload.get("sub")
        user_groups: List[str] = payload.get("user_groups", ["public"])
        exp: float = payload.get("exp")
        
        if username is None:
            return None
            
        # Check token expiration
        if datetime.utcnow().timestamp() > exp:
            logger.warning(f"Token expired for user: {username}")
            return None
            
        return TokenPayload(sub=username, user_groups=user_groups, exp=exp)
    except JWTError as e:
        logger.error(f"JWT verification failed: {e}")
        return None
