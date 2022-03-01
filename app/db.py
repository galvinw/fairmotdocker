# app/db.py

import pydantic
import databases
import ormar
import sqlalchemy
from config import settings
from pydantic import Json
from datetime import datetime

database = databases.Database(settings.db_url)
metadata = sqlalchemy.MetaData()

class BaseModel(ormar.Model):
    class Meta:
        abstract = True
        metadata = metadata
        database = database
    
    id         : int        = ormar.Integer  (primary_key=True)
    created_at : datetime   = ormar.DateTime (default=datetime.now)
    updated_at : datetime   = ormar.DateTime (default=None, nullable=True)
    is_deleted : bool       = ormar.Boolean  (default=False, nullable=False)

class User(BaseModel):
    class Meta(ormar.ModelMeta):
        tablename = "user"

    username   : str    = ormar.String  (max_length=128, unique=True, nullable=False)
    is_active  : bool   = ormar.Boolean (default=True, nullable=False)

class Camera(BaseModel):
    class Meta(ormar.ModelMeta):
        tablename = "camera"

    name                : str   = ormar.String  (max_length=128,nullable=False, default="Camera")
    connection_string   : str   = ormar.String  (max_length=256,nullable=False)
    # threshold           : int   = ormar.Integer (nullable=True)
    # lat                 : float = ormar.Float   (nullable=True)
    # long                : float = ormar.Float   (nullable=True)
    # camera_shift_time   : int   = ormar.Integer (default=0)
    position_x          : float = ormar.Float   (nullable=True)
    position_y          : float = ormar.Float   (nullable=True)
    position_z          : float = ormar.Float   (nullable=True)
    focal_length        : float = ormar.Float   (nullable=True)
    # lastchange          : datetime = ormar.DateTime(default=datetime.now) 

class Person(BaseModel):
    class Meta(ormar.ModelMeta):
        tablename = "person"

    strack_id   : int   = ormar.Integer (index=True, nullable=False)
    name        : str   = ormar.String  (max_length=128,nullable=False, default="Person")
    is_active   : bool  = ormar.Boolean (default=True, nullable=False)

class PersonInstance(BaseModel):
    class Meta(ormar.ModelMeta):
        tablename = "person_instance"

    strack_id   : int   = ormar.Integer (index=True, nullable=False)
    name        : str   = ormar.String  (max_length=128,nullable=False)
    camera_id   : int   = ormar.Integer (nullable=True)                  # Not for production
    frame_id    : int   = ormar.Integer (nullable=True)                  # Not for production
    conf_level  : float = ormar.Float   (nullable=True)
    status      : str   = ormar.String  (max_length=128,nullable=True)
    position_x  : float = ormar.Float   (nullable=True)
    position_z  : float = ormar.Float   (nullable=True)

class Zone(BaseModel):
    class Meta(ormar.ModelMeta):
        tablename = "zone"

    name          : str  = ormar.String  (max_length=128,nullable=False, default="Zone")
    camera_id     : int  = ormar.Integer (nullable=False, default=1)
    coordinates   : Json = ormar.JSON    (nullable=False)

# class ZoneStatus(BaseModel):
#     class Meta(ormar.ModelMeta):
#         tablename = "zone_status"

#     zone_id               : str = ormar.Integer(nullable=False, default=0)

class PersonZoneStatus(BaseModel):
    class Meta(ormar.ModelMeta):
        tablename = "person_zone_status"

    strack_id   : int = ormar.Integer   (nullable=False)
    person_name : str = ormar.String    (max_length=128,nullable=False)
    zone_id     : int = ormar.Integer   (index=True, nullable=False)


engine = sqlalchemy.create_engine(settings.db_url)
metadata.create_all(engine)


class RequestUser(pydantic.BaseModel):
    class Config:
        orm_mode = True

    username    : str

class RequestCamera(pydantic.BaseModel):
    class Config:
        orm_mode = True

    name                : str
    connection_string   : str
    # threshold           : int
    # lat                 : float
    # long                : float
    # camera_shift_time   : int
    position_x          : float
    position_y          : float
    position_z          : float
    focal_length        : float


class RequestPerson(pydantic.BaseModel):
    class Config:
        orm_mode = True

    strack_id   : int
    name        : str

class RequestPersonInstance(pydantic.BaseModel):
    class Config:
        orm_mode = True

    strack_id   : int  
    name        : str  
    camera_id   : int  
    frame_id    : int  
    conf_level  : float
    status      : str  
    position_x  : float
    position_z  : float

class RequestPersonZoneStatus(pydantic.BaseModel):
    class Config:
        orm_mode = True

    strack_id   : int
    person_name : str
    zone_id     : int

class RequestZone(pydantic.BaseModel):
    class Config:
        orm_mode = True

    name          : str  
    camera_id     : int  
    coordinates   : Json 

class RequestMonofair(pydantic.BaseModel):
    class Config:
        orm_mode = True

    strack_id   : int  
    conf_level  : float
    status      : str  
    position_x  : float
    position_z  : float

class RequestFrame(pydantic.BaseModel):
    class Config:
        orm_mode = True

    camera_id   : int  
    frame_id    : int  