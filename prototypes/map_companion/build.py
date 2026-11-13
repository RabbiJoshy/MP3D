"""Refresh the isolated preview from the promoted app without changing the release."""
from pathlib import Path
here = Path(__file__).resolve().parent
source = here.parents[1] / 'docs' / 'index.html'
s = source.read_text()
s = s.replace('<title>MP3D —', '<title>MP3D · Map companion preview —')
s = s.replace('localStorage.getItem(k)', "localStorage.getItem('map-companion:'+k)").replace('localStorage.setItem(k,v)', "localStorage.setItem('map-companion:'+k,v)")
(here / 'index.html').write_text(s)
print('Refreshed isolated preview:', here / 'index.html')
