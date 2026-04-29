import random
import numpy as np
import math
from .mesh import Mesh
from .car_mesh import CarMesh

# ── Car colors (dark palette) ────────────────────────────────────────────────
CAR_COLORS = [
    (0.0,  0.30, 0.45),   # dark teal / cyan
    (0.05, 0.28, 0.05),   # dark green
    (0.55, 0.45, 0.0 ),   # dark yellow / gold
    (0.50, 0.05, 0.05),   # dark red
    (0.50, 0.10, 0.30),   # dark pink / magenta
]

# Road boundaries (from Plane vertices)
PLANE_X_MIN, PLANE_X_MAX = -15.3, 3.95
PLANE_Z_MIN, PLANE_Z_MAX = -12.21, 2.75

# Lane centres (Z) and travel direction (dir=+1 → +X, -1 → -X)
LANES = [
    {'z': -3.5, 'dir':  1},
    {'z': -5.5, 'dir': -1},
]

GROUND_Y    = -2.505
CAR_SCALE   = 0.8
TREE_SCALE  = 0.5
LAMP_SCALE  = 0.15   # lamppost is ~14 units tall, building ~5 → scale down heavily

CAR_SPEED = 2.0      # units / second


class CarState:
    """Runtime state for one animated car."""
    def __init__(self, car_id, obj_id, lane, x_start, color):
        self.car_id  = car_id
        self.obj_id  = obj_id
        self.lane    = lane
        self.x       = x_start
        self.color   = color
        self.wheel_angle = 0.0   # radians – accumulated rotation

    def update(self, dt):
        speed = CAR_SPEED * self.lane['dir']
        self.x += speed * dt
        # wheel radius ≈ 0.3 units in scaled model; ω = v / r
        wheel_radius = 0.3 * CAR_SCALE
        self.wheel_angle += (abs(speed) / wheel_radius) * dt

    @property
    def out_of_bounds(self):
        return self.x > PLANE_X_MAX + 2 or self.x < PLANE_X_MIN - 2


class SceneManager:
    def __init__(self):
        self.models   = {}
        self.objects  = []
        self.car_states: list[CarState] = []
        self._next_obj_id = 100
        self._color_idx   = 0

    # ── Loading ───────────────────────────────────────────────────────────────
    def load_models(self):
        print("Loading models…")
        self.models['building'] = Mesh('object/building/build.obj')
        self.models['car']      = CarMesh('object/car/model.obj')   # group-aware
        self.models['tree']     = Mesh('object/tree/Tree.obj')
        self.models['lamppost'] = Mesh('object/lamppost/lamppost.obj')
        print("Models loaded.")

    # ── Scene generation (called once; cars are managed separately) ───────────
    def generate_scene(self, seed=None):
        if seed is not None:
            random.seed(seed)

        self.objects.clear()
        self.car_states.clear()
        self._next_obj_id = 10

        # Building
        self.objects.append(self._make_static('building', [0,0,0], [0,0,0], [1,1,1], obj_id=1))

        # ── Static trees ────────────────────────────────────────────────────
        # 4 trees in deleted-building slot (Buil.001: X -3.23→-8.14, Z -2.45→2.45)
        for pos in [(-4.0, GROUND_Y,  1.0),
                    (-6.0, GROUND_Y,  1.5),
                    (-5.5, GROUND_Y, -1.5),
                    (-7.5, GROUND_Y,  0.0)]:
            self._add_tree(pos)

        # 4 random sidewalk trees
        for _ in range(4):
            x = random.uniform(PLANE_X_MIN + 1, PLANE_X_MAX - 1)
            z = random.choice([random.uniform(-12.0, -9.5),
                               random.uniform(1.5, 2.5)])
            self._add_tree((x, GROUND_Y, z))

        # ── Lampposts ───────────────────────────────────────────────────────
        # The lamppost arm (Torus.001) extends in local +X direction.
        # R_y(θ) maps local +X → world (cos θ, 0, −sin θ).
        # To aim arm at world direction (dx, dz): θ = atan2(−dz, dx).
        def lamp_yaw(dx, dz):
            return math.atan2(-dz, dx)

        # Park-style: 4 corners, arms pointing toward deleted-building centre (~-5.685, 0)
        park_cx, park_cz = -5.685, 0.0
        park_corners = [
            (-3.8,  2.0),   # NE corner
            (-7.6,  2.0),   # NW corner
            (-3.8, -2.0),   # SE corner
            (-7.6, -2.0),   # SW corner
        ]
        for (lx, lz) in park_corners:
            dx = park_cx - lx
            dz = park_cz - lz
            self._add_lamp([lx, GROUND_Y, lz], lamp_yaw(dx, dz))

        # Road-edge: 2 lampposts at Z=-2.5, arms perpendicular to car travel (aim at -Z, into road)
        # dx=0, dz=-1  →  θ = atan2(-(-1), 0) = π/2
        road_yaw = math.atan2(1.0, 0.0)   # π/2
        for lx in [-4.5, -6.5]:
            self._add_lamp([lx, GROUND_Y, -2.5], road_yaw)

        # ── Initial cars (spread evenly, no overlaps) ─────────────────────
        for i in range(4):
            lane   = LANES[i % 2]
            # Space cars ~6 units apart on each lane
            offset = (i // 2) * 7.0
            x0 = PLANE_X_MIN + 4 + offset if lane['dir'] == 1 else PLANE_X_MAX - 4 - offset
            if self._can_spawn(lane, x0):
                self._spawn_car(lane, x0)

        return self.objects

    # ── Car animation ─────────────────────────────────────────────────────────
    def update(self, dt):
        """Call every frame with delta-time in seconds."""
        to_remove = []
        for cs in self.car_states:
            cs.update(dt)
            if cs.out_of_bounds:
                to_remove.append(cs)

        for cs in to_remove:
            self.car_states.remove(cs)
            self.objects = [o for o in self.objects if o.get('car_id') != cs.car_id]
            # Spawn replacement at the entry end of the lane
            lane = cs.lane
            x_start = PLANE_X_MIN - 1 if lane['dir'] == 1 else PLANE_X_MAX + 1
            if self._can_spawn(lane, x_start):
                self._spawn_car(lane, x_start)

        # Sync car object positions from states
        for obj in self.objects:
            if obj.get('type') != 'car':
                continue
            cid = obj.get('car_id')
            cs  = next((s for s in self.car_states if s.car_id == cid), None)
            if cs:
                obj['position'][0] = cs.x
                obj['wheel_angle'] = cs.wheel_angle

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _make_static(self, type_, pos, rot, scl, obj_id=None):
        oid = obj_id if obj_id is not None else self._new_id()
        return {
            'type': type_,
            'mesh': self.models[type_],
            'position': list(pos),
            'rotation': list(rot),
            'scale': list(scl),
            'id': oid,
        }

    def _add_tree(self, pos):
        rot_y = random.uniform(0, 2*math.pi)
        self.objects.append({
            'type': 'tree',
            'mesh': self.models['tree'],
            'position': list(pos),
            'rotation': [0, rot_y, 0],
            'scale': [TREE_SCALE]*3,
            'id': self._new_id(),
        })

    def _add_lamp(self, pos, yaw):
        self.objects.append({
            'type': 'lamppost',
            'mesh': self.models['lamppost'],
            'position': list(pos),
            'rotation': [0, yaw, 0],
            'scale': [LAMP_SCALE]*3,
            'id': self._new_id(),
        })

    def _can_spawn(self, lane, x_start, min_gap=5.0):
        """Return True only if no existing car is closer than min_gap on the same lane."""
        for cs in self.car_states:
            if cs.lane['z'] == lane['z'] and abs(cs.x - x_start) < min_gap:
                return False
        return True

    def _spawn_car(self, lane, x_start):
        color = CAR_COLORS[self._color_idx % len(CAR_COLORS)]
        self._color_idx += 1
        car_id  = self._new_id()
        obj_id  = self._new_id()
        yaw     = math.radians(90 if lane['dir'] == 1 else -90)
        cs = CarState(car_id, obj_id, lane, x_start, color)
        self.car_states.append(cs)
        self.objects.append({
            'type':  'car',
            'mesh':  self.models['car'],
            'position':    [x_start, GROUND_Y, lane['z']],
            'rotation':    [0, yaw, 0],
            'scale':       [CAR_SCALE]*3,
            'id':          obj_id,
            'car_id':      car_id,
            'color':       color,
            'wheel_angle': 0.0,
        })

    def _new_id(self):
        v = self._next_obj_id
        self._next_obj_id += 1
        return v
