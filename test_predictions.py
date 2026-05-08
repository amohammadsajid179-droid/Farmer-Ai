import httpx, json

API = 'http://localhost:8000'

tests = [
    {'name': 'Rice conditions',   'data': {'N':80,'P':45,'K':42,'temperature':23,'humidity':82,'ph':6.5,'rainfall':230,'season':'Kharif','soil_type':'Clayey','farm_area':2}},
    {'name': 'Wheat conditions',  'data': {'N':80,'P':48,'K':42,'temperature':21,'humidity':79,'ph':6.8,'rainfall':175,'season':'Rabi','soil_type':'Loamy','farm_area':2}},
    {'name': 'Lentil conditions', 'data': {'N':12,'P':16,'K':16,'temperature':25,'humidity':59,'ph':6.0,'rainfall':38,'season':'Rabi','soil_type':'Sandy','farm_area':1}},
    {'name': 'Cotton conditions', 'data': {'N':82,'P':18,'K':24,'temperature':24,'humidity':89,'ph':6.8,'rainfall':148,'season':'Kharif','soil_type':'Sandy','farm_area':3}},
    {'name': 'Mango conditions',  'data': {'N':31,'P':112,'K':202,'temperature':32,'humidity':64,'ph':6.4,'rainfall':59,'season':'Zaid','soil_type':'Sandy','farm_area':1}},
    {'name': 'Coffee conditions', 'data': {'N':24,'P':133,'K':201,'temperature':29,'humidity':89,'ph':5.7,'rainfall':98,'season':'Zaid','soil_type':'Loamy','farm_area':1}},
    {'name': 'Apple conditions',  'data': {'N':101,'P':81,'K':51,'temperature':17,'humidity':64,'ph':6.4,'rainfall':77,'season':'Rabi','soil_type':'Loamy','farm_area':1}},
    {'name': 'Coconut conditions','data': {'N':57,'P':30,'K':36,'temperature':22,'humidity':91,'ph':5.5,'rainfall':178,'season':'Kharif','soil_type':'Sandy','farm_area':1}},
]

print('=== CROP PREDICTION TEST RESULTS ===')
for t in tests:
    r = httpx.post(f'{API}/predict', json=t['data'])
    if r.status_code == 200:
        recs = r.json()['recommendations'][:3]
        top  = recs[0]
        name = t['name']
        crop = top['crop']
        conf = top['confidence']
        c2   = recs[1]['crop']
        c3   = recs[2]['crop']
        print(f'  {name:22s} -> TOP: {crop:14s} ({conf:.1f}%) | 2nd: {c2} | 3rd: {c3}')
    else:
        print(f'  ERROR {r.status_code}: {r.text[:100]}')

print()
# Health check
h = httpx.get(f'{API}/health')
print('Health:', h.json())
print('[DONE] All tests complete!')
