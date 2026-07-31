# SO-ARM100/SO101 — lamp head conversion

`so101_new_calib.xml` and `scene.xml` are the real SO-ARM100/SO101 model
(Apache-2.0), vendored from
[TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100)
(`UPSTREAM_README.md` is their original readme). Modified in two ways:

## 1. Gripper -> lamp head, still 6 DOF

The upstream model's gripper is two parts: the **`gripper` body** (holds the
`wrist_roll` joint — despite the name, this is a wrist-positioning joint, not
part of the jaw) and a **`moving_jaw_so101_v1` body** (child of `gripper`,
holds the `gripper` joint — this is the actual jaw open/close DOF, and its
`moving_jaw_so101_v1` mesh geom is the visible claw finger).

Only the jaw finger reads as "gripper," so only that mesh geom was removed.
The joint and body it lived on were kept and repurposed:

| MJCF name (unchanged) | Was | Now |
|---|---|---|
| body `gripper` | wrist-roll housing | unchanged; also carries the fixed `lamp_neck` geom |
| joint `wrist_roll` | wrist roll DOF | unchanged |
| body `moving_jaw_so101_v1` | jaw base | carries `lamp_shade` + `lamp_bulb` + `lamp_spotlight` |
| joint `gripper` | jaw open/close | **repurposed as `head_tilt`** — same joint, same range, tilts the shade instead of a jaw |

**Joint/body/actuator names were deliberately left as `gripper`/`wrist_roll`
in the XML** rather than renamed to `head_tilt` — this is a compatibility
choice, not an oversight: `behavior/ik.py`'s `_CHAIN_BODIES` list references
the body name `"gripper"` directly, and nothing in the Python code reads MJCF
joint names at all (`behavior/motor.py`'s `DEFAULT_JOINT_LIMITS` and every
`move_to()` call address joints positionally, by list index). Renaming would
have touched more files for zero functional benefit. Channel index 5 (the
last of the 6-vector every `move_to()` call takes) is semantically
`head_tilt` for lamp purposes — this table is the source of truth for that
mapping, not the XML.

**Result: still 6 DOF.** `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex,
wrist_roll` position the head (unchanged, still the 5 joints `ik.py`'s
`LampIK.solve()` returns); `head_tilt` (ex-`gripper`) tilts the shade
independently and is not part of the position IK chain, same as the jaw
never was.

## 2. Lamp head assembly geometry

Three new pieces. Positioned using each body's *empirically measured* axis
(computed from `body_xmat`/`site_xpos` at the home pose and cross-checked
against `light_xdir`, not assumed — a first pass used the wrong axis for the
shade, pointing it sideways instead of down; caught by rendering, not by eye
alone, then fixed by explicitly solving for which local axis maps to world
`(0,0,-1)`):

- **Neck** (`lamp_neck`, plain cylinder, no mesh needed): child geom of body
  `gripper`. Cylinders are symmetric, so only the position matters: offset
  along `gripper`'s local **-Z** (a fixed mechanical bridge toward
  `moving_jaw_so101_v1`, not related to which way the shade points). 30mm
  long, 15mm radius.
- **Shade** (`lamp_shade`, `assets/lamp_shade.obj` — a generated frustum, open
  top and bottom so the bulb/glow is visible from an angle; MuJoCo has no
  native truncated-cone primitive, see `scripts/gen_lamp_shade_mesh.py`; 60mm
  tall, 55mm opening / 30mm neck-end diameter): child geom of body
  `moving_jaw_so101_v1`. The mesh is authored with its axis along mesh-local
  +Z (narrow end at the neck attachment); this body's **local -X**, not -Y,
  is the axis that points straight down at the home pose (verified: the
  resulting `light_xdir` for the co-located spotlight comes out to
  `(0, 0.05, -0.999)`, i.e. ~2.8° off vertical), so the geom's
  `quat="0.70710678 0 -0.70710678 0"` (a -90° rotation about Y) maps mesh+Z
  onto body -X.
- **Bulb** (`lamp_bulb`, 12mm sphere, `emission="1.5"`) + **halo** (larger,
  low-alpha sphere for a soft glow) + **spotlight** (`lamp_spotlight`, MuJoCo
  `<light>`): co-located, recessed near the shade's narrow (top) interior —
  `pos="-0.015 0 0"`, i.e. 15mm in along the same -X axis, well short of the
  60mm-deep shade, so it's enclosed rather than poking out the wide opening.
  The light's `dir="-1 0 0"` matches the shade's opening axis exactly (both
  derived from the same -X finding).

**Verified kinematically** (`bulb_site`, a site co-located with the bulb, plus
`d.light_xdir[lamp_spotlight]` for the direction claim specifically): moving
`shoulder_pan` alone moves the whole assembly (measured position deltas, not
just visual inspection); moving `head_tilt` alone swings the bulb while
leaving the base position essentially unchanged; running `look_at()` moves
the bulb toward the target (small residual offset is expected — `look_at`
positions the `gripperframe` site, which sits a few cm from `bulb_site`, not
the bulb itself); a top-down render confirms the spotlight produces a visible
illuminated patch on the floor beneath the shade.

## Scene (`scene.xml`)

Floor is now a warm wood-brown checker (`reflectance=0.05`, was a blue-gray
checker); skybox is a cream-to-soft-gray gradient (was blue); a backdrop
plane removes the infinite-floor horizon; a 3-light rig (key/fill/ambient)
lights the scene independent of the lamp's own spotlight. A named `demo_cam`
(`mode="targetbody"`, aimed at `moving_jaw_so101_v1` — MuJoCo computes the aim
direction natively, deliberately avoiding hand-derived look-at axes after the
shade-orientation bug above) gives a reproducible close, low, front-on
framing that actually shows the downward-pointing shade's profile — but note
**`behavior/motor.py`'s live render loop uses the free/default camera, not a
named one** (not touched, per instruction), so `<statistic center extent>`
and `<visual><global azimuth elevation>` were separately tuned (azimuth 240,
elevation 15 — empirically matched to `demo_cam`'s framing, not derived) so
the *default* camera also frames the lamp head closely. If a specific named
camera should drive the live render instead, that's a one-line change in
`MuJoCoMotorBackend._render_loop`'s `update_scene()` call — flagged, not made.
