"""sk20 fetch: all EDMs since 2024-07-04 from official Parliament API; cached to disk."""
import json, time, pathlib, urllib.request, ssl, certifi
CTX=ssl.create_default_context(cafile=certifi.where())
R=pathlib.Path("/Users/joshuathomasamar/PycharmProjects/MP3D"); D=R/"data/quick/research_2026_11_10/sk20/edm"
B="https://oralquestionsandmotions-api.parliament.uk"
def get(url):
    req=urllib.request.Request(url,headers={"User-Agent":"mp3d-research-script"})
    for t in range(4):
        try: return urllib.request.urlopen(req,timeout=60,context=CTX).read().decode()
        except Exception as e: time.sleep(2+3*t); err=e
    raise err
lst=[]; skip=0
f=D/"list.json"
if f.exists(): lst=json.loads(f.read_text())
else:
    while True:
        j=json.loads(get(f"{B}/EarlyDayMotions/list?parameters.tabledStartDate=2024-07-04&parameters.skip={skip}&parameters.take=100"))
        r=j["Response"]; lst+=r; skip+=100; time.sleep(0.2)
        if not r or skip>=j["PagingInfo"]["Total"]: break
    f.write_text(json.dumps(lst))
print("list",len(lst),flush=True)
(D/"det").mkdir(exist_ok=True)
from concurrent.futures import ThreadPoolExecutor
def one(e):
    p=D/"det"/f"{e['Id']}.json"
    if p.exists(): return
    p.write_text(get(f"{B}/EarlyDayMotion/{e['Id']}")); time.sleep(0.2)
with ThreadPoolExecutor(5) as ex:
    for i,_ in enumerate(ex.map(one,lst)):
        if i%300==0: print(i,flush=True)
print("done")
