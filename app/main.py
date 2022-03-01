# app/main.py

import json

from typing import List
from fastapi import FastAPI, BackgroundTasks
from datetime import datetime
from db import database, User, Camera, Person, PersonInstance, Zone, PersonZoneStatus 
from db import RequestUser, RequestCamera, RequestPerson, RequestPersonInstance, RequestPersonZoneStatus, RequestZone, RequestMonofair, RequestFrame
from utils import read_json

tags_metadata = [
    {"name": "Users", "description": ""},
    {"name": "Cameras", "description": ""},
    {"name": "Persons", "description": ""},
    {"name": "Person Instances", "description": ""},
    {"name": "Person Zone Status", "description": ""},
    {"name": "Zones", "description": ""},
]

app = FastAPI(title="Lauretta Built Environment Analytics", openapi_tags=tags_metadata)

@app.get("/")
async def read_root():
    return "Welcome to Lauretta Built Environment Analytics"

@app.get("/users/", response_model=List[User], tags=["Users"])
async def read_users():
    return await User.objects.all()

@app.post("/users/", response_model=User, tags=["Users"])
async def create_user(user: RequestUser):
    return await User.objects.get_or_create(username=user.username)

@app.get("/cameras/", response_model=List[Camera], tags=["Cameras"])
async def read_cameras():
    return await Camera.objects.all()

@app.get("/cameras/name/{name}", response_model=Camera, tags=["Cameras"])
async def read_camera_name(name: str):
    return await Camera.objects.exclude(is_deleted=True).get_or_none(name=name)

@app.post("/cameras/", response_model=Camera, tags=["Cameras"])
async def create_camera(camera: RequestCamera):
    return await Camera.objects.get_or_create(
        name=camera.name,
        connection_string=camera.connection_string,
        position_x=camera.position_x,
        position_y=camera.position_y,
        position_z=camera.position_z,
        # threshold=camera.threshold,
        # lat=camera.lat,
        # long=camera.long,
        # camera_shift_time=camera.camera_shift_time,
        focal_length=camera.focal_length)

@app.get("/persons/", response_model=List[Person], tags=["Persons"])
async def read_all_person():
    return await Person.objects.exclude(is_deleted=True).all()

@app.get("/persons/strack-id/{strack_id}", response_model=Person, tags=["Persons"])
async def read_person_id(strack_id: int):
    return await Person.objects.exclude(is_deleted=True).get_or_none(strack_id=strack_id)

@app.get("/persons/name/{name}", response_model=Person, tags=["Persons"])
async def read_person_name(name: str):
    return await Person.objects.exclude(is_deleted=True).get_or_none(name=name)

@app.get("/persons/active", response_model=List[Person], tags=["Persons"])
async def read_all_active_person():
    return await Person.objects.filter(is_active='yes').exclude(is_deleted=True).all()

@app.post("/persons/", response_model=Person, tags=["Persons"])
async def create_active_person(person: RequestPerson):
    return await Person.objects.get_or_create(
        strack_id=person.strack_id,
        name=person.name)

@app.post("/persons/active/{strack_id}", response_model=Person, tags=["Persons"])
async def reactivate_person(strack_id: int):
    person = await Person.objects.exclude((Person.is_deleted == True) | (Person.is_active == True)).get_or_none(strack_id=strack_id)
    if (person):
        return await person.update(is_active=True, updated_at=datetime.now())

@app.post("/persons/inactive/{strack_id}", response_model=Person, tags=["Persons"])
async def inactivate_person(strack_id: int):
    person = await Person.objects.exclude((Person.is_deleted == True) | (Person.is_active == False)).get_or_none(strack_id=strack_id)
    if (person):
        return await person.update(is_active=False, updated_at=datetime.now())

@app.post("/persons/delete/{strack_id}", response_model=Person, tags=["Persons"])
async def delete_person(strack_id: int):
    person = await Person.objects.exclude(is_deleted=True).get_or_none(strack_id=strack_id)
    if (person):
        return await person.update(is_deleted=True, updated_at=datetime.now())

@app.post("/persons/undelete/{strack_id}", response_model=Person, tags=["Persons"])
async def undelete_person(strack_id: int):
    person = await Person.objects.exclude(is_deleted=False).get_or_none(strack_id=strack_id)
    if (person):
        return await person.update(is_deleted=False, updated_at=datetime.now())

@app.post("/person-instances/", response_model=PersonInstance, tags=["Person Instances"])
async def create_person_instance(person_instance: RequestPersonInstance):
    return await PersonInstance.objects.create(
        strack_id=person_instance.strack_id,
        name=person_instance.name,
        camera_id=person_instance.camera_id, 
        frame_id=person_instance.frame_id,
        conf_level=person_instance.conf_level,
        status=person_instance.status, 
        position_x=person_instance.position_x, 
        position_z=person_instance.position_z)

@app.get("/person-instances/", response_model=List[PersonInstance], tags=["Person Instances"])
async def read_person_instance():
    return await PersonInstance.objects.all()

@app.get("/person-zone-status/", response_model=List[PersonZoneStatus], tags=["Person Zone Status"])
async def read_all_person_zone_status():
    return await PersonZoneStatus.objects.all()
@app.get("/person-zone-status/person-name/{person_name}", response_model=PersonZoneStatus, tags=["Person Zone Status"])
async def read_person_zone_status_by_name(person_name: str):
    return await PersonZoneStatus.objects.get_or_none(person_name=person_name)

@app.get("/person-zone-status/zone-id/{zone_id}", response_model=PersonZoneStatus, tags=["Person Zone Status"])
async def read_person_zone_status_by_zid(zone_id: int):
    return await PersonZoneStatus.objects.get_or_none(zone_id=zone_id)

@app.get("/person-zone-status/strack-id/{strack_id}", response_model=PersonZoneStatus, tags=["Person Zone Status"])
async def read_person_zone_status_by_sid(strack_id: int):
    return await PersonZoneStatus.objects.get_or_none(strack_id=strack_id)

@app.post("/person-zone-status/", response_model=PersonZoneStatus, tags=["Person Zone Status"])
async def create_person_zone_status(person_zone: RequestPersonZoneStatus):
    return await PersonZoneStatus.objects.get_or_create(
        strack_id=person_zone.strack_id,
        person_name=person_zone.person_name, 
        zone_id=person_zone.zone_id)

@app.get("/zones/", response_model=List[Zone], tags=["Zones"])
async def read_all_zones():
    return await Zone.objects.all()

@app.get("/zones/name/{name}", response_model=Zone, tags=["Zones"])
async def read_zone_name(name: str):
    return await Zone.objects.exclude(is_deleted=True).get_or_none(name=name)

@app.post("/zones/", response_model=Zone, tags=["Zones"])
async def create_zone(zone: RequestZone):
    name = zone.name
    zone_in_db = await read_zone_name(name)
    if not (zone_in_db):
        # Unable to use get_or_create due to json type for coordinates
        return await Zone.objects.create(
            name=name,
            camera_id=zone.camera_id,
            coordinates=zone.coordinates)

@app.post("/monofair/", response_model=PersonInstance, tags=["Person Instances"])
async def process_monofair_dic_out(dic_out: RequestMonofair, frame_info: RequestFrame):
    strack_id = dic_out.strack_id
    person = await read_person_id(strack_id)
    if (person):
        name = person.name
    else:
        name = "Person Not Found"

    # Pseudocode for person zone_status
    '''
    If (x, z) is within zone:
        person_zone = RequestPersonZoneStatus(   )
        create_person_zone_status(person_zone)
    '''

    person_instance = RequestPersonInstance(
        strack_id=strack_id,
        name=name,
        camera_id=frame_info.camera_id,
        frame_id=frame_info.frame_id,
        conf_level=dic_out.conf_level,
        status=dic_out.status,
        position_x=dic_out.position_x,
        position_z=dic_out.position_z
    )

    return await create_person_instance(person_instance)

@app.on_event("startup")
async def startup():
    if not database.is_connected:
        await database.connect()
        await post_cameras("/config/camera.json")
        await post_zones("/config/zone.json")

@app.on_event("shutdown")
async def shutdown():
    if database.is_connected:
        await database.disconnect()


async def post_cameras(file_path):
    data = read_json(file_path)

    for cam in data["camera_list"]:
        camera = RequestCamera(
            name=cam["camera_name"],
            connection_string=cam["connection_string"],
            position_x=cam["position_x"],
            position_y=cam["position_y"],
            position_z=cam["position_z"],
            focal_length=cam["focal_length"]
        )
        await create_camera(camera)

async def post_zones(file_path):
    data = read_json(file_path)

    for zone_data in data["zone_list"]:
        camera_name = zone_data["camera_name"]
        camera = await read_camera_name(camera_name)

        coord = zone_data["coordinates"]
        assert len(coord["position_x"]) == len(coord["position_z"]), "Position X should have the same number of coordinates with Position Z"

        zone = RequestZone(
            name=zone_data["zone_name"],
            camera_id=camera.id,
            coordinates=json.dumps(coord)
        )
        await create_zone(zone)
