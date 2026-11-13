# Map companion interface

The interface is now part of the canonical app. `companion.css` and `companion.js` are embedded by `build_hierarchy_app.py` through placeholders in `hierarchy_app_template.html`.

The layout preserves the map data, positions, source evidence, quiz and relationship explorer. It adds contextual topic panels, visible dimension/axis controls, sidebar toggling, bounded wheel/button zoom, topic-marker settings and a two-section guide.

To refresh the isolated preview from the release:

    python3 prototypes/map_companion/build.py

Preview: http://127.0.0.1:8765/prototypes/map_companion/index.html

The preview uses isolated browser-storage keys. Updating it does not modify the release. Canonical release validation: `node prototypes/test_release.cjs`.
