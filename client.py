import datetime
import requests
import numpy as np
import json

addr = 'http://localhost:8181'
test_url = addr + '/predict'
content_type = 'application/json'
headers = {'content-type': content_type}

temp = np.zeros((2, 4)) + 0.1
temp = temp.tolist()
s = ''
while True:
    s = input("enter:")
    data = {'data': s}

    t = datetime.datetime.now()
    response = requests.post(test_url, json=json.dumps(data), headers=headers)
    t2 = datetime.datetime.now()
    #print(t2-t)
    j2 = json.loads(response.text)
    print(str(j2['agm']) + '\t' + j2['best_p'])
