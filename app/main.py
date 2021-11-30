# app/main.py

from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from datetime import datetime
from db import database, User, Zones, Cameras, PersonInstance, Person, Zone_Status
import json
# from src import track

app = FastAPI(title="Lauretta Built Environment Analytics")
# fair = FastAPI()



class PersonRequest(BaseModel):
    name: str

class ZoneStatusRequest(BaseModel):
    create_at: datetime
    zone_id: int
    number: int


@app.get("/")
async def read_root():
    return await User.objects.all()

@app.get("/users/")
async def read_users():
    return await User.objects.all()

@app.get("/zone_status/" )
async def read_zone_status(zoneid: int = 1):
    return await Zone_Status.objects.filter(zone_id = zoneid).order_by(Zone_Status.create_at.desc()).limit(1).all()

@app.get("/cameras/")
async def read_cameras():
    return await Cameras.objects.all()

@app.get("/person_instance/")
async def read_person_instance():
    return await PersonInstance.objects.all()

@app.get("/person/")
async def read_person():
    return await Person.objects.all()


@app.post("/add_zone_status/")
async def update_zone_status(zone_status: Zone_Status):
    zone_json = zone_status.json()
    zone_dict = json.loads(zone_json)

    await Zone_Status.objects.create(zone_id=int(zone_dict['zone_id']),number=int(zone_dict['number']))
    return zone_dict

@app.post("/add_person/", response_model=Person)
async def add_person(person: Person):
    person_json = person.json()
    person_dict = json.loads(person_json)

    await Person.objects.get_or_create(name=person_dict['name'])
    return person_dict

@app.post("/add_person_instance/")
async def add_person_instance(person_instance: PersonInstance):
    person_instance_json = person_instance.json()
    person_instance_dict = json.loads(person_instance_json)

    # await PersonInstance.objects.create(name=person_instance_dict['name'],x=float(person_instance_dict['x']),z=float(person_instance_dict['z']))
    await PersonInstance.objects.create(name=person_instance_dict['name'],frame_id=int(person_instance_dict['frame_id']))
    return person_instance_dict

# @app.patch("/patch_person_instance/{frame_id}/{name}")
@app.patch("/patch_person_instance/")
async def patch_person_instance(person_instance: PersonInstance):
    person_instance_json = person_instance.json()
    person_instance_dict = json.loads(person_instance_json)

    await PersonInstance.objects.update(name=person_instance_dict['name'],frame_id=int(person_instance_dict['frame_id']),x=float(person_instance_dict['x']),z=float(person_instance_dict['z']))
    return person_instance_dict

@app.on_event("startup")
async def startup():
    if not database.is_connected:
        await database.connect()
        # create a dummy entry
        await User.objects.get_or_create(email='test@email.com')
        await Zones.objects.get_or_create(name="Zone 1")
        await Cameras.objects.get_or_create(name="Camera 1",connectionstring='TestVideo17.mp4')
        await camerareader()
        await zonereader()
        await PersonInstance.objects.get_or_create(name="PersonInstance0")
        await Person.objects.get_or_create(name="Person 0")
        


@app.on_event("shutdown")
async def shutdown():
    if database.is_connected:
        await database.disconnect()



# @fair.on_event("startup")
# async def startup():
#     await FairMOT

# app.mount("/fair", fair)


async def camerareader():

    f = open("/config/cameras.txt", "r")
    camera_list = f.readlines()
    f.close()
    await Cameras.objects.delete(each=True)

    print("Camera CSV Length: {}",str(len(camera_list)))

    for element in camera_list:
        element = element.split(",")
        print(element)

        await Cameras.objects.get_or_create(name=element[0],connectionstring=element[1],threshold=int(element[2]),lat=float(element[3]),long=float(element[4]))
    print("Camera CSV Read")

async def zonereader():

    f = open("/config/zones.txt", "r")
    zone_list = f.readlines()
    f.close()
    print("Zone CSV Length: {}",str(len(zone_list)))
    await Zones.objects.delete(each=True)
    for element in zone_list:
        element = element.split(",")
        print(element)

        await Zones.objects.get_or_create(name=element[0],camera_id=int(element[1]),x=int(element[2]),y=int(element[3]))
    print("Zone CSV Read")


# async def FairMOT():
#     track.eval_prop()
#     print("Running FairMOT Tracker")
