# app/db.py

from email.policy import default
import databases
import ormar
import sqlalchemy
from datetime import datetime

# from .config import settings

from pydantic import BaseSettings, Field

class Settings(BaseSettings):
    db_url: str = Field(..., env='DATABASE_URL')

settings = Settings()

database = databases.Database(settings.db_url)
metadata = sqlalchemy.MetaData()

class BaseMeta(ormar.ModelMeta):
    metadata = metadata
    database = database

class User(ormar.Model):
    class Meta(BaseMeta):
        tablename = "user"
    id: int = ormar.Integer(primary_key=True)
    username: str = ormar.String(max_length=128, unique=True, nullable=False)
    is_active: bool = ormar.Boolean(default=True, nullable=False)
    is_deleted: bool = ormar.Boolean(default=False, nullable=False)
    created_at: datetime = ormar.DateTime(default=datetime.now)
    updated_at: datetime = ormar.DateTime(default=None, nullable=True)

class Camera(ormar.Model):
    class Meta(BaseMeta):
        tablename = "camera"
    id: int = ormar.Integer(primary_key=True)
    name: str = ormar.String(max_length=128,nullable=False, default="Camera")
    connectionstring: str = ormar.String(max_length=256,nullable=False)
    threshold: int = ormar.Integer(nullable=True)
    lat:float = ormar.Float(nullable=True)
    long:float = ormar.Float(nullable=True)
    camera_shift_time: int = ormar.Integer( default=0)
    focal_length:float = ormar.Float(nullable=True)
    # lastchange: datetime = ormar.DateTime(default=datetime.now) 
    is_deleted: bool = ormar.Boolean(default=False, nullable=False)
    created_at: datetime = ormar.DateTime(default=datetime.now)
    updated_at: datetime = ormar.DateTime(default=None, nullable=True)

class Person(ormar.Model):
    class Meta(BaseMeta):
        tablename = "person"
    id: int = ormar.Integer(primary_key=True)
    name: str = ormar.String(max_length=128,nullable=False, default="Person")
    is_active: bool = ormar.Boolean(default=True, nullable=False)
    is_deleted: bool = ormar.Boolean(default=False, nullable=False)
    created_at: datetime = ormar.DateTime(default=datetime.now)
    updated_at: datetime = ormar.DateTime(default=None, nullable=True)

class PersonInstance(ormar.Model):
    class Meta(BaseMeta):
        tablename = "person_instance"
    id: int = ormar.Integer(primary_key=True)
    name: str = ormar.String(max_length=128,nullable=False)
    camera_id: int = ormar.Integer(nullable=True)       # Not for production
    frame_id: int = ormar.Integer(nullable=True)        # Not for production
    conf_level:float = ormar.Float(nullable=True)
    status: str = ormar.String(max_length=128,nullable=True)
    position_x:float = ormar.Float(nullable=True)
    position_z:float = ormar.Float(nullable=True)
    is_deleted: bool = ormar.Boolean(default=False, nullable=False)
    created_at: datetime = ormar.DateTime(default=datetime.now)
    updated_at: datetime = ormar.DateTime(default=None, nullable=True)

class Zone(ormar.Model):
    class Meta(BaseMeta):
        tablename = "zone"
    id: int = ormar.Integer(primary_key=True)
    name: str = ormar.String(max_length=128,nullable=False, default="Zone")
    camera_id: int = ormar.Integer(nullable=False, default=1)
    zone_x1: int = ormar.Integer(nullable=False, default=0)
    zone_y1: int = ormar.Integer(nullable=False, default=0)
    zone_x2: int = ormar.Integer(nullable=False, default=0)
    zone_y2: int = ormar.Integer(nullable=False, default=0)
    is_deleted: bool = ormar.Boolean(default=False, nullable=False)
    created_at: datetime = ormar.DateTime(default=datetime.now)
    updated_at: datetime = ormar.DateTime(default=None, nullable=True)

class ZoneStatus(ormar.Model):
    class Meta(BaseMeta):
        tablename = "zone_status"
    id: int = ormar.Integer(primary_key=True)
    zone_id: str = ormar.Integer(nullable=False, default=0)
    is_deleted: bool = ormar.Boolean(default=False, nullable=False)
    created_at: datetime = ormar.DateTime(default=datetime.now)
    updated_at: datetime = ormar.DateTime(default=None, nullable=True)

class PersonZoneStatus(ormar.Model):
    class Meta(BaseMeta):
        tablename = "person_zone_status"
    id: int = ormar.Integer(primary_key=True)
    name: str = ormar.String(max_length=128,nullable=False)
    zone_id: str = ormar.Integer(nullable=False)
    is_deleted: bool = ormar.Boolean(default=False, nullable=False)
    created_at: datetime = ormar.DateTime(default=datetime.now)
    updated_at: datetime = ormar.DateTime(default=None, nullable=True)


engine = sqlalchemy.create_engine(settings.db_url)
metadata.create_all(engine)