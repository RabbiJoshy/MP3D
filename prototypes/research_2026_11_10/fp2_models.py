import re, json, pathlib, requests
ROOT = pathlib.Path("/Users/joshuathomasamar/PycharmProjects/MP3D")
key = next(m[1].strip().strip("\"'") for l in (ROOT / ".env").read_text().splitlines() if (m := re.match(r"\s*(?:export\s+)?DEEPINFRA_API_KEY\s*=\s*(.*)", l)))
x = requests.get("https://api.deepinfra.com/v1/openai/models", headers={"Authorization": f"Bearer {key}"}, timeout=60); print(x.status_code)
d = x.json()["data"]; json.dump(d, open(ROOT/"data/quick/research_2026_11_10/fp2/models.json","w"))
print(len(d)); print(json.dumps(d[0])[:600])
