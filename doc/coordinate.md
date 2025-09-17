Multiple coordinate systems are used:
- Gate's global (world) system
- Allpix's local (sensor) system (see https://allpix-squared.docs.cern.ch/docs/05_geometry_detectors/01_geometry/)

To transform coordinates between systems:
- See coordinate_transform.jpg
- Use localFractional2globalCoordinates / global2localFractionalCoordinates in utils.py

