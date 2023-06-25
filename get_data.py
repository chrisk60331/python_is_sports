# Python 3
import http.client, urllib.parse
import os
import time
ACCESS_KEY = os.environ.get('mediastack_api_key')
conn = http.client.HTTPConnection('api.mediastack.com')
CATS = ["sports,-business", "business,-sports"]
params = {
    'access_key': ACCESS_KEY,
    'limit': 100,
    'languages': 'en',
}
for cat in CATS:
    params['categories'] = cat
    for indx in range(0, 1000, params.get('limit', 100)):
        params['offset'] = indx
        encoded_params = urllib.parse.urlencode(params)
        conn.request('GET', '/v1/news?{}'.format(encoded_params))
        res = conn.getresponse()
        data = res.read().decode('utf-8')
        target = f"{cat}{indx}.json"
        open(target, "w").write(data)
        time.sleep(3)
        print(f"wrote {len(data)} to {target}")
