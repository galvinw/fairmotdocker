# app/db.py

import databases
import ormar
import sqlalchemy
from datetime import datetime

from config import settings

database = databases.Database(settings.db_url)
metadata = sqlalchemy.MetaData()


class BaseModel(ormar.Model):
    class Meta:
        abstract = True
        metadata = metadata
        database = database
    
    id: int = ormar.Integer(primary_key=True)
    created_at: datetime = ormar.DateTime(default=datetime.now)
    updated_at: datetime = ormar.DateTime(default=None, nullable=True)
    is_deleted: bool = ormar.Boolean(default=False, nullable=False)

class User(BaseModel):
    class Meta(ormar.ModelMeta):
        tablename = "user"

    username: str = ormar.String(max_length=128, unique=True, nullable=False)
    is_active: bool = ormar.Boolean(default=True, nullable=False)

class Camera(BaseModel):
    class Meta(ormar.ModelMeta):
        tablename = "camera"

    name: str = ormar.String(max_length=128,nullable=False, default="Camera")
    connectionstring: str = ormar.String(max_length=256,nullable=False)
    threshold: int = ormar.Integer(nullable=True)
    lat:float = ormar.Float(nullable=True)
    long:float = ormar.Float(nullable=True)
    camera_shift_time: int = ormar.Integer( default=0)
    focal_length:float = ormar.Float(nullable=True)
    # lastchange: datetime = ormar.DateTime(default=datetime.now) 

class Person(BaseModel):
    class Meta(ormar.ModelMeta):
        tablename = "person"

    name: str = ormar.String(max_length=128,nullable=False, default="Person")
    is_active: bool = ormar.Boolean(default=True, nullable=False)

class PersonInstance(BaseModel):
    class Meta(ormar.ModelMeta):
        tablename = "person_instance"

    name: str = ormar.String(max_length=128,nullable=False)
    camera_id: int = ormar.Integer(nullable=True)       # Not for production
    frame_id: int = ormar.Integer(nullable=True)        # Not for production
    conf_level:float = ormar.Float(nullable=True)
    status: str = ormar.String(max_length=128,nullable=True)
    position_x:float = ormar.Float(nullable=True)
    position_z:float = ormar.Float(nullable=True)

class Zone(BaseModel):
    class Meta(ormar.ModelMeta):
        tablename = "zone"

    name: str = ormar.String(max_length=128,nullable=False, default="Zone")
    camera_id: int = ormar.Integer(nullable=False, default=1)
    zone_x1: int = ormar.Integer(nullable=False, default=0)
    zone_y1: int = ormar.Integer(nullable=False, default=0)
    zone_x2: int = ormar.Integer(nullable=False, default=0)
    zone_y2: int = ormar.Integer(nullable=False, default=0)

class ZoneStatus(BaseModel):
    class Meta(ormar.ModelMeta):
        tablename = "zone_status"

    zone_id: str = ormar.Integer(nullable=False, default=0)

class PersonZoneStatus(BaseModel):
    class Meta(ormar.ModelMeta):
        tablename = "person_zone_status"

    name: str = ormar.String(max_length=128,nullable=False)
    zone_id: str = ormar.Integer(nullable=False)


engine = sqlalchemy.create_engine(settings.db_url)
metadata.create_all(engine)