"""Run a paid fp2 stage under a shared, conservative $2 request budget.
Usage: python fp2_budget.py stage.py [args...]
Prices are the pipeline's recorded provider rates, not a billing reconciliation.
Unknown/failed requests retain their reservation; successful usage replaces it.
"""
import json, pathlib, sqlite3, sys, runpy, threading
import requests
ROOT = pathlib.Path(__file__).resolve().parents[2]
DB = ROOT / 'data/quick/research_2026_11_10/fp2/budget.sqlite'
LIMIT = 2.0
lock = threading.Lock()
with sqlite3.connect(DB) as db:
    db.execute('CREATE TABLE IF NOT EXISTS spend (id INTEGER PRIMARY KEY, stage TEXT, usd REAL, status TEXT)')
    if not db.execute('SELECT 1 FROM spend').fetchone():
        db.execute('INSERT INTO spend(stage,usd,status) VALUES (?,?,?)', ('completed extraction', .1813, 'reported'))
original = requests.Session.post

def post(self, url, **kwargs):
    body = kwargs.get('json', {})
    if 'typesafe.ai/' in url:
        price_in, price_out = .042, 0
        # Conservative UTF-8 byte bound, includes both questions and templates.
        bound_in = 2 * (len(json.dumps(body, ensure_ascii=False).encode()) + 4096)
        bound_out = 0
    elif 'deepinfra.com/' in url:
        price_in, price_out = {'deepseek-ai/DeepSeek-V3.2': (.26, .38), 'Qwen/Qwen3-235B-A22B-Instruct-2507': (.09, .55)}.get(body.get('model'), (.03, .15))
        bound_in = len(json.dumps(body, ensure_ascii=False).encode()) + 4096
        bound_out = body.get('max_tokens', 300)
    else:
        raise RuntimeError('Unbudgeted provider')
    reserved = (bound_in * price_in + bound_out * price_out) / 1e6
    with lock, sqlite3.connect(DB, timeout=60, isolation_level=None) as db:
        db.execute('BEGIN IMMEDIATE')
        total = db.execute('SELECT SUM(usd) FROM spend').fetchone()[0] or 0
        if total + reserved > LIMIT:
            raise SystemExit(f'Budget ceiling: ${total:.4f} committed; request withheld')
        rid = db.execute('INSERT INTO spend(stage,usd,status) VALUES (?,?,?)', (pathlib.Path(sys.argv[0]).name, reserved, 'reserved')).lastrowid
    response = original(self, url, **kwargs)
    if response.status_code in (401,402,403):
        raise SystemExit(f'Provider authorization/quota failure ({response.status_code}); stopping')
    if response.status_code == 200:
        usage = response.json().get('usage', {})
        inp = usage.get('input_tokens', usage.get('prompt_tokens'))
        out = usage.get('completion_tokens', 0)
        if inp is not None:
            usd = (inp * price_in + out * price_out) / 1e6
            with lock, sqlite3.connect(DB, timeout=60) as db:
                db.execute('UPDATE spend SET usd=?,status=? WHERE id=?', (usd, 'reported', rid))
    return response

if __name__ == '__main__':
    requests.Session.post = post
    stage = pathlib.Path(sys.argv[1]).resolve()
    sys.argv = [str(stage), *sys.argv[2:]]
    sys.path.insert(0, str(stage.parent))
    try:
        runpy.run_path(str(stage), run_name='__main__')
    finally:
        with sqlite3.connect(DB) as db:
            print('Budget ledger USD:', round(db.execute('SELECT SUM(usd) FROM spend').fetchone()[0], 5), flush=True)
