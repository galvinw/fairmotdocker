
def txtreader():

    f = open("/config/cameras.txt", "r")
    camera_list = f.readlines()
    f.close()

    for element in itertools.cycle(camera_list):
        print(element)
        element = element.split(",")
        cameraName = element[0]
        cameraIP = element[1]
        threshold = element[2]
        lat = element[3]
        longi = element[4]
        camera_shift_time = int(element[6])
        prev_time = time.time()

    return 0