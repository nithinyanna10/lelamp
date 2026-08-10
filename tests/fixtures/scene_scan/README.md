Drop 2-3 real photos here (e.g. `bottle.jpg`, `laptop.jpg`, `mug.jpg`) with an
entry in `EXPECTED_CLASSES` in `tests/test_scene_scan.py` for each, to exercise
`run_yolo_world` against real images locally. Empty by default -- the test
skips itself when there's nothing here, so CI stays green without a model
download or committed binary images.
