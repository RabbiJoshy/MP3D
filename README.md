# MP3D — Parliament’s space

Explore MPs, subjects and arguments in a 3D sphere, read the speeches behind recorded positions, and answer the same questions to compare yourself with MPs.

## Current app

The canonical source is `prototypes/build_hierarchy_app.py` plus `prototypes/hierarchy_app_template.html`, with `ha_quiz.js` and `ha_relationships.js`. It includes checked asks, source excerpts, 633 searchable MP profiles, topic navigation, axis explanations and the quiz. The sphere and agreed topic hierarchy are preserved.

```sh
.venv/bin/python prototypes/build_hierarchy_app.py
node prototypes/test_release.cjs
python3 -m http.server 8765 --bind 127.0.0.1
```

Open [the local app](http://127.0.0.1:8765/docs/index.html). Building writes identical copies to `docs/index.html`, `docs/stance.html`, `prototypes/app_hierarchy.html` and the old `prototypes/sk13_app_asks.html` path. The sk13 builder is a compatibility entry point, not an alternative implementation. Only a separate Git push publishes the docs files.

The build uses existing local data and cached models; the gitignored research evidence is required. See `HANDOVER.md` for dependencies, verification and outstanding work.

## Interpretation

Profiles report extracted speech positions with supporting excerpts; they can still contain errors. Unknown positions are not evidence of neutrality. The global map compresses topic stance scores into three dimensions. Topic planes are approximate summaries, and the visitor star is a rough weighted-direction placement rather than an exact encoding of every answer. Direct comparisons use shared observed answers. Focus data exists separately but is not yet integrated into this release.

The previous unsupervised topic-axis app remains in `docs/faultlines.html`; older experiments remain in `prototypes/`. The legacy builders no longer write the current release pages.

Data: UK Parliament and TheyWorkForYou/ParlParse. Parliamentary information is licensed under the Open Parliament Licence.
