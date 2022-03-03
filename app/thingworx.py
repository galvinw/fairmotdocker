import requests
from urllib3.exceptions import InsecureRequestWarning

PUSH_TO_THINGWORX = 'YES'         # 'y' or 'yes' for yes (case insensitive), any other values will be no

THINGWORX_BASE_URL = 'https://35.187.239.24:8443'
THINGWORX_APP_KEY = '260be65c-151e-4035-a9a7-44674cce2b62'
THINGWORX_THING = 'LaurettaThing'

def check_twx():
    twx = PUSH_TO_THINGWORX.lower()
    if twx == 'y' or twx == 'yes':
        return True

def put_twx_properties(prop: str, data=None):
    if not (check_twx()):
        return None

    url = f"{THINGWORX_BASE_URL}/Thingworx/Things/{THINGWORX_THING}/Properties/{prop}"
    headers = {
        "Content-Type": "application/json", 
        "appKey": THINGWORX_APP_KEY
    }
    data = {prop: data.json()}

    try:
        # NOTE: For production, verify=True should be used to check for server certificates
        res = requests.put(url, json=data, headers=headers, verify=False)

        print(f"PUT/ {url}")
        print(f"Response = {res}")
        return res
    except Exception as e:
        print(e)
        print(f"Failed to PUT/ {url}")

def twx_post_camera(cam):
    prop = "Camera"
    return put_twx_properties(prop, cam)

def twx_post_zone(person_zone):
    # NOTE: Zone name in db and Thingworx must be the same (case sensitive)
    prop = person_zone.zone_name
    return put_twx_properties(prop, person_zone)


def main():
    if check_twx():
        print("Data will be pushed to Thingworx")

        # Suppress the following warning since our requests does not check validity of certificates:
            # InsecureRequestWarning: Unverified HTTPS request is being made to host '35.187.239.24'. Adding certificate verification is strongly advised. See: https://urllib3.readthedocs.io/en/1.26.x/advanced-usage.html#ssl-warnings
        # NOTE: For production, this should be commented
        requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)
    else:
        print("Data will not be pushed to Thingworx")

main()
