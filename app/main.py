# app/main.py

from turtle import update
from typing import List
from fastapi import FastAPI, BackgroundTasks
from pydantic import Json
from datetime import datetime
from db import database, User, Camera, Person, PersonInstance, Zone, PersonZoneStatus 
from db import RequestUser, RequestCamera, RequestPerson, RequestPersonInstance, RequestPersonZoneStatus, RequestZone

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

@app.post("/cameras/", response_model=Camera, tags=["Cameras"])
async def create_camera(camera: RequestCamera):
    return await Camera.objects.get_or_create(
        name=camera.name,
        connectionstring=camera.connectionstring,
        threshold=camera.threshold,
        lat=camera.lat,
        long=camera.long,
        camera_shift_time=camera.camera_shift_time,
        focal_length=camera.focal_length)

@app.get("/persons/", response_model=List[Person], tags=["Persons"])
async def read_all_person():
    return await Person.objects.all()

@app.get("/persons/id/{id}", response_model=Person, tags=["Persons"])
async def read_person_id(id: int):
    return await Person.objects.get_or_none(id=id)

@app.get("/persons/name/{name}", response_model=Person, tags=["Persons"])
async def read_person_name(name: str):
    return await Person.objects.get_or_none(name=name)

@app.get("/persons/active", response_model=List[Person], tags=["Persons"])
async def read_all_active_person():
    return await Person.objects.filter(is_active='yes').all()

@app.post("/persons/", response_model=Person, tags=["Persons"])
async def create_active_person(person: RequestPerson):
    return await Person.objects.get_or_create(
        strack_id=person.strack_id,
        name=person.name)

@app.post("/persons/active/{id}", response_model=Person, tags=["Persons"])
async def reactivate_person(id: int):
    person = await Person.objects.get_or_none(id=id)
    if person is None:
        return None
    elif person.is_active==True:
        return person
    else:
        now = datetime.now()
        return await person.update(is_active=True, updated_at=now)

@app.post("/persons/inactive/{id}", response_model=Person, tags=["Persons"])
async def inactivate_person(id: int):
    person = await Person.objects.get_or_none(id=id)
    if person is None:
        return None
    elif person.is_active==False:
        return person
    else:
        now = datetime.now()
        return await person.update(is_active=False, updated_at=now)

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

@app.post("/zones/", response_model=Zone, tags=["Zones"])
async def create_zone(zone: RequestZone):
    return await Zone.objects.create(
        name=zone.name,
        camera_id=zone.camera_id,
        coordinates=zone.coordinates)

# @app.get("/zone_status/" )
# async def read_zone_status(zoneid: int = 1):
#     return await ZoneStatus.objects.filter(zone_id = zoneid).order_by(ZoneStatus.create_at.desc()).limit(1).all()

# @app.post("/add_zone_status/")
# async def update_zone_status(zone_status: ZoneStatus):
#     zone_json = zone_status.json()
#     zone_dict = json.loads(zone_json)

#     await ZoneStatus.objects.create(zone_id=int(zone_dict['zone_id']),number=int(zone_dict['number']))
#     return zone_dict

@app.on_event("startup")
async def startup():
    if not database.is_connected:
        await database.connect()
        # create a dummy entry
        # await User.objects.get_or_create(username='test@email.com')
        # await Zone.objects.get_or_create(name="Zone 1")
        # await Camera.objects.get_or_create(name="Camera 1",connectionstring='TestVideo17.mp4')
        # await camerareader()
        # await zonereader()
        # await PersonInstance.objects.get_or_create(name="PersonInstance0")
        # await Person.objects.get_or_create(name="Person 0")
        


@app.on_event("shutdown")
async def shutdown():
    if database.is_connected:
        await database.disconnect()


# async def camerareader():

#     f = open("/config/cameras.txt", "r")
#     camera_list = f.readlines()
#     f.close()
#     await Camera.objects.delete(each=True)

#     print("Camera CSV Length: {}",str(len(camera_list)))

#     for element in camera_list:
#         element = element.split(",")
#         print(element)

#         await Camera.objects.get_or_create(name=element[0],connectionstring=element[1],threshold=int(element[2]),lat=float(element[3]),long=float(element[4]))
#     print("Camera CSV Read")

# async def zonereader():

#     f = open("/config/zones.txt", "r")
#     zone_list = f.readlines()
#     f.close()
#     print("Zone CSV Length: {}",str(len(zone_list)))
#     await Zone.objects.delete(each=True)
#     for element in zone_list:
#         element = element.split(",")
#         print(element)

#         await Zone.objects.get_or_create(name=element[0],camera_id=int(element[1]),zone_x1=int(element[2]),zone_y1=int(element[3]),zone_x2=int(element[4]),zone_y2=int(element[5]))
#     print("Zone CSV Read")

