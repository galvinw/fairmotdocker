import requests

BASE_URL = 'http://web:8000'

def post(url, data=None):
    try:
        res = requests.post(url, json=data, headers={"content-type":"application/json"})
        print(f"Successful POST/ {url}")
        print(f"Response = {res}")
    except Exception as e:
        print(e)
        print(f"Failed to POST/ {url}")


def create_person(strack_id):
    person_obj = {
        "strack_id": strack_id,
        "name": f"Person {strack_id}"
    }
    post(f"{BASE_URL}/persons/", person_obj)

def inactivate_person(strack_id):
    post(f"{BASE_URL}/persons/inactive/{strack_id}")
    
def reactivate_person(strack_id):
    post(f"{BASE_URL}/persons/active/{strack_id}")

def delete_person(strack_id):
    post(f"{BASE_URL}/persons/delete/{strack_id}")
    
def create_person_instances(monofair_dic_out, camera_id, frame_id):
    total_person = monofair_dic_out["total_person"]
    active_person_ids = monofair_dic_out["active_person_ids"]
    ious = monofair_dic_out["ious"]
    status = monofair_dic_out["status"]
    xyz_preds = monofair_dic_out["xyz_preds"]


    if total_person > 0 and len(xyz_preds) > 0:
        for idx, id in enumerate(active_person_ids):

            x = xyz_preds[idx][0]
            # y = xyz_preds[idx][1]
            z = xyz_preds[idx][2]


            monofair_obj = {
                "dic_out": {
                    "strack_id": id,
                    "conf_level": ious[idx],
                    "status": status[idx],
                    "position_x": x,
                    "position_z": z
                },
                "frame_info": {
                    "camera_id": camera_id,
                    "frame_id": frame_id
                }
            }

            post(f"{BASE_URL}/monofair/", monofair_obj)