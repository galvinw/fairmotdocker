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