"""Group 5 uses a compass, three remembered places, and mixed rides on currents.

Spring and summer are for finding and holding food. In fall, jellyfish with
offspring capacity head for a nursery. Every zone boundary crossing pins that
zone's center one radius from the crossing point. Switching currents day by day
gives any average velocity that a mix of the four currents allows, so a
jellyfish can steer along headings no single current offers. When the currents
cannot cancel out, a jellyfish times its arrival and rides the slowest mix
through the zone instead of holding still. If danger overlaps the destination,
the jellyfish aims for the safer side.
"""

from functools import lru_cache
from itertools import combinations
from math import atan2, ceil, cos, floor, hypot, inf, pi, sin, tau

from core.action import Action
from core.constants import OCEAN_SIZE, ZONE_RADIUS
from core.depth import Depth
from core.player import Player
from core.seasons import Season
from core.snapshot import JellyfishSnapshot

# Distances are in km and velocities in km/day. Memory stores sites on a grid.
_HALF = OCEAN_SIZE / 2
_R = ZONE_RADIUS
_FOOD, _NURSERY, _DANGER = 2, 4, 6
_WINDOWS = {_FOOD: (91, 180), _NURSERY: (271, 360)}  # summer, winter
_DEPTHS = tuple(Depth)
_SEARCH_DEPTHS = (Depth.SURFACE, Depth.SHALLOW, Depth.MEDIUM)
_SECTORS = 15  # Compass sectors per nibble; 0 means not sampled yet.
_X_CELL = OCEAN_SIZE / 256

_HOLD_DRIFT = 1.0  # Below this unavoidable drift (km/day) a jellyfish can hold still.
_HORIZON = 3  # Days used to steer back to a station.
_ENTRY_DEPTH = pi / 4 * _R  # Mean distance ahead to the center after crossing in.
_MAX_NUDGE = 25.0  # km a single consistency check may move an estimate.
_ARRIVED = _R / 2  # km; this deep in the estimated target yet outside: estimate wrong.
_ANGLE_BINS = 360
_DANGER_BERTH = 1.1  # Keep this many radii from the estimated predator center.
_ESCAPE_MARGIN = 10.0  # km; when escaping, assume we are at least this far inside.
_ESCAPE_SWING = pi * 40 / 180  # How far off the predator estimate may be.

Point = tuple[float, float]


def _wrap(offset: float) -> float:
    """Shortest signed displacement along one axis of the ocean."""
    return (offset + _HALF) % OCEAN_SIZE - _HALF


def _distance_sq(point: Point, x: float, y: float) -> float:
    dx = _wrap(point[0] - x)
    dy = _wrap(point[1] - y)
    return dx * dx + dy * dy


def _velocity(depth: Depth, sector: int) -> Point:
    angle = tau * (sector - 1) / _SECTORS
    return depth.speed * sin(angle), -depth.speed * cos(angle)


# ---------------------------------------------------------------- mixing currents
# Riding current a for some days and current b for others averages to a point on
# the segment between their velocities. All mixes of four currents form the
# convex hull of the four velocity points.


def _closest_on_segment(p: Point, a: Point, b: Point) -> Point:
    dx, dy = b[0] - a[0], b[1] - a[1]
    length_sq = dx * dx + dy * dy
    t = 0.0
    if length_sq > 0:
        t = max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / length_sq))
    return a[0] + t * dx, a[1] + t * dy


def _in_triangle(p: Point, a: Point, b: Point, c: Point) -> bool:
    def side(u: Point, v: Point) -> float:
        return (v[0] - u[0]) * (p[1] - u[1]) - (v[1] - u[1]) * (p[0] - u[0])

    area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    if abs(area) < 1e-9:
        return False
    s1, s2, s3 = side(a, b), side(b, c), side(c, a)
    return (s1 >= 0 and s2 >= 0 and s3 >= 0) or (s1 <= 0 and s2 <= 0 and s3 <= 0)


def _mix_closest(p: Point, velocities: tuple[Point, ...]) -> Point:
    """The achievable average velocity closest to p."""
    if any(_in_triangle(p, *tri) for tri in combinations(velocities, 3)):
        return p
    return min(
        (_closest_on_segment(p, a, b) for a, b in combinations(velocities, 2)),
        key=lambda q: (q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2,
    )


def _top_speed(ux: float, uy: float, velocities: tuple[Point, ...]) -> float:
    """Fastest average speed along the unit heading (ux, uy); 0 if unreachable."""
    best = 0.0
    for a, b in combinations(velocities, 2):
        ex, ey = b[0] - a[0], b[1] - a[1]
        det = ex * uy - ux * ey
        if abs(det) < 1e-12:
            continue
        # Solve s * u = a + t * (b - a) for s >= 0 and 0 <= t <= 1.
        s = (ex * a[1] - ey * a[0]) / det
        t = (ux * a[1] - uy * a[0]) / det
        if -1e-9 <= t <= 1 + 1e-9:
            best = max(best, s)
    return best


@lru_cache(maxsize=32)
def _mixes(compass: int) -> tuple[tuple[Point, ...], Point, tuple[float, ...], Point]:
    """Velocities, slowest mix, top speed per heading and a central mix."""
    velocities = tuple(_velocity(d, compass >> (4 * d) & 15) for d in _DEPTHS)
    slowest = _mix_closest((0.0, 0.0), velocities)
    top = tuple(
        _top_speed(cos(tau * i / _ANGLE_BINS), sin(tau * i / _ANGLE_BINS), velocities)
        for i in range(_ANGLE_BINS)
    )
    middle = (
        sum(v[0] for v in velocities) / 4,
        sum(v[1] for v in velocities) / 4,
    )
    return velocities, slowest, top, middle


def _travel_time(x: float, y: float, top: tuple[float, ...]) -> float:
    distance = hypot(x, y)
    if distance < 1e-9:
        return 0.0
    speed = top[round(atan2(y, x) / tau * _ANGLE_BINS) % _ANGLE_BINS]
    return distance / speed if speed > 0 else inf


def _fastest_copy(goal: Point, top: tuple[float, ...]) -> tuple[Point, float]:
    """The copy of goal on the wrapped ocean that a mix of currents reaches soonest."""
    gx, gy = _wrap(goal[0]), _wrap(goal[1])
    best, best_time = (gx, gy), inf
    for reach in (1, 2):
        for i in range(-reach, reach + 1):
            for j in range(-reach, reach + 1):
                x, y = gx + i * OCEAN_SIZE, gy + j * OCEAN_SIZE
                time = _travel_time(x, y, top)
                if time < best_time:
                    best, best_time = (x, y), time
        if best_time < inf:
            break
    return best, best_time


def _on_time_copy(
    goal: Point, days: int, velocities: tuple[Point, ...], middle: Point
) -> Point | None:
    """The nearest copy of goal reachable in exactly `days` days, if any."""
    gx, gy = _wrap(goal[0]), _wrap(goal[1])
    cx = (days * middle[0] - gx) / OCEAN_SIZE
    cy = (days * middle[1] - gy) / OCEAN_SIZE
    best, best_length = None, inf
    for i in {floor(cx), ceil(cx)}:
        for j in {floor(cy), ceil(cy)}:
            x, y = gx + i * OCEAN_SIZE, gy + j * OCEAN_SIZE
            wx, wy = x / days, y / days
            mx, my = _mix_closest((wx, wy), velocities)
            if hypot(wx - mx, wy - my) < 0.05 and hypot(x, y) < best_length:
                best, best_length = (x, y), hypot(x, y)
    return best


# ---------------------------------------------------------------- the map


def _pin_to_rim(x: float, y: float, px: float, py: float) -> Point:
    """We crossed a boundary at (px, py), so the center is one radius from there."""
    dx, dy = x - px, y - py
    distance = hypot(dx, dy)
    if distance == 0:
        return x, y
    return px + dx * _R / distance, py + dy * _R / distance


def _nudge(x: float, y: float, inside: bool) -> Point:
    """Without a crossing, keep the estimate consistent with today's reading."""
    distance = hypot(x, y)
    if distance == 0 or inside == (distance <= _R):
        return x, y
    # Inside puts the center within one radius; outside puts it farther away.
    step = max(-_MAX_NUDGE, min(_MAX_NUDGE, _R - distance))
    scale = 1 + step / distance
    return x * scale, y * scale


def _observe_currents(s: JellyfishSnapshot) -> tuple[int, int, int, int]:
    sector = 1 + round(s.heading * _SECTORS / 360) % _SECTORS
    slot, half = divmod(s.depth, 2)
    shift = 4 * half
    s.memory[slot] = (s.memory[slot] & ~(15 << shift)) | (sector << shift)
    first, second = s.memory[0], s.memory[1]
    return first & 15, first >> 4, second & 15, second >> 4


def _danger_cost(danger: Point | None, x: float, y: float) -> float:
    if danger is None:
        return 0.0
    # We saw part of a predator zone. Give that spot a berth smaller than its radius.
    return 3 * max(0.0, (_DANGER_BERTH * _R) ** 2 - _distance_sq(danger, x, y))


def _days_inside(vx: float, vy: float, cx: float, cy: float) -> float:
    """Days from now spent within one radius of (cx, cy) while riding (vx, vy)."""
    # Solve |v t - c| = R for t.
    speed_sq = vx * vx + vy * vy
    toward = vx * cx + vy * cy
    gap = cx * cx + cy * cy - _R * _R
    root = toward * toward - speed_sq * gap
    if root <= 0:
        return 0.0
    enter = (toward - root**0.5) / speed_sq
    leave = (toward + root**0.5) / speed_sq
    return max(0.0, leave - max(enter, 0.0))


def _risk(danger: Point | None, vx: float, vy: float) -> float:
    """Danger cost of riding (vx, vy) for one day, or for three."""
    return max(_danger_cost(danger, vx, vy), _danger_cost(danger, 3 * vx, 3 * vy))


class Player5(Player):
    """Spend summer at food and spend winter capacity at a safe nursery.

    Each jellyfish keeps its compass in bytes 0..1. Bytes 2..7 hold the
    relative positions of food, nursery and danger, two bytes each. One bit per
    zone remembers whether the jellyfish was inside it yesterday, so a change
    marks a boundary crossing. With 9+ bytes those bits live in byte 8; with
    exactly 8 they take the lowest bit of each site's y byte, which halves the
    y resolution. Offspring inherit all of it from their parent.
    """

    def __init__(self, memory_size, lifetime_days, rng):
        super().__init__(memory_size, lifetime_days, rng)
        self._flags_in_y = memory_size < 9
        self._y_levels = 128 if self._flags_in_y else 256
        self._y_cell = OCEAN_SIZE / self._y_levels

    # ------------------------------------------------------------ memory codec
    def _read_site(self, memory: bytearray, site: int) -> Point | None:
        x_byte, y_byte = memory[site], memory[site + 1]
        if self._flags_in_y:
            y_byte >>= 1
        if x_byte == y_byte == 0:
            return None
        return (x_byte - 128) * _X_CELL, (y_byte - self._y_levels // 2) * self._y_cell

    def _write_site(self, memory: bytearray, site: int, x: float, y: float) -> None:
        # Round at random so repeated fractional steps do not bias the map.
        x_byte = (128 + floor(x / _X_CELL + self.rng.random())) % 256
        y_byte = (
            self._y_levels // 2 + floor(y / self._y_cell + self.rng.random())
        ) % self._y_levels
        if x_byte == y_byte == 0:
            x_byte = 1  # Keep (0, 0) available for an undiscovered site.
        memory[site] = x_byte
        if self._flags_in_y:
            memory[site + 1] = (y_byte << 1) | (memory[site + 1] & 1)
        else:
            memory[site + 1] = y_byte

    def _clear_site(self, memory: bytearray, site: int) -> None:
        memory[site] = 0
        memory[site + 1] &= 1 if self._flags_in_y else 0

    def _was_inside(self, memory: bytearray, site: int) -> bool:
        if self._flags_in_y:
            return bool(memory[site + 1] & 1)
        return bool(memory[8] >> (site // 2 - 1) & 1)

    def _set_inside(self, memory: bytearray, site: int, inside: bool) -> None:
        if self._flags_in_y:
            memory[site + 1] = (memory[site + 1] & ~1) | inside
        else:
            bit = 1 << (site // 2 - 1)
            memory[8] = (memory[8] | bit) if inside else (memory[8] & ~bit)

    # ------------------------------------------------------------ decisions
    def act(self, s: JellyfishSnapshot) -> Action:
        breeding = (
            s.season in (Season.FALL, Season.WINTER) and s.reproduction_capacity > 0
        )
        inside = s.in_breeding_zone if breeding else s.in_feeding_zone
        if len(s.memory) < 8:
            return self._simple(s, breeding, inside)

        headings = _observe_currents(s)
        target = _NURSERY if breeding else _FOOD
        self._update_map(s, target)
        danger = self._read_site(s.memory, _DANGER)
        if s.in_predator_zone:
            return Action(self._escape(headings, danger))

        # Sample each layer once. The currents then stay fixed for the whole run.
        if 0 in headings:
            return Action(Depth(headings.index(0)))
        center = self._read_site(s.memory, target)
        if center is None:
            return self._explore(s)
        mixes = _mixes(s.memory[0] | s.memory[1] << 8)
        depth = self._steer(s.day_of_year, target, center, danger, mixes)
        return self._explore(s) if depth is None else Action(depth)

    def _steer(
        self,
        day: int,
        target: int,
        center: Point,
        danger: Point | None,
        mixes: tuple[tuple[Point, ...], Point, tuple[float, ...], Point],
    ) -> Depth | None:
        velocities, slowest, top, middle = mixes
        speed = hypot(*slowest)
        drift = speed >= _HOLD_DRIFT
        station = self._station(center, danger, slowest if drift else None)
        start, end = _WINDOWS[target]
        in_season = start <= day <= end

        goal = station
        if drift:
            # We cannot hold still, so ride the slowest mix along a line through
            # the station, entering at the upstream rim as the season opens.
            ux, uy = slowest[0] / speed, slowest[1] / speed
            sx, sy = station
            offset = hypot(sx - center[0], sy - center[1])
            half = 0.9 * max(0.0, _R * _R - offset * offset) ** 0.5
            lead = min(half, speed * (end - start) / 2)
            entry = (sx - ux * lead, sy - uy * lead)
            along = -(sx * ux + sy * uy)  # Our position along the line.
            goal = entry
            if in_season and -lead <= along <= half:
                # Already on the patch: keep riding from here, never back upstream.
                goal = (
                    sx + ux * along + slowest[0] * _HORIZON,
                    sy + uy * along + slowest[1] * _HORIZON,
                )

        near = (_wrap(goal[0]), _wrap(goal[1]))
        if hypot(*near) <= _R:
            # Close by: hold around the goal, even when looping around the ocean
            # would technically reach it sooner.
            return self._track(near, _HORIZON, velocities, danger)
        (gx, gy), time = _fastest_copy(goal, top)
        if time == inf:
            return None
        days_left = (start - day) % 360
        if not in_season and drift and time < days_left:
            # Arriving early would just drift past, so arrive as the season opens.
            on_time = _on_time_copy(goal, days_left, velocities, middle)
            if on_time is not None:
                return self._track(on_time, days_left, velocities, danger)
        if time <= _HORIZON:
            return self._track((gx, gy), _HORIZON, velocities, danger)
        if hypot(gx, gy) > _R:
            waypoint = self._detour((gx, gy), danger)
            if waypoint != (gx, gy) and _travel_time(*waypoint, top) < inf:
                gx, gy = waypoint

        # Far away: take the current that leaves the least travel time after today.
        # Near ties alternate between the currents that bracket the heading, which
        # mixes them into a heading that no single current offers.
        best_depth, best_score = None, inf
        for depth, (vx, vy) in zip(_DEPTHS, velocities):
            score = _travel_time(gx - vx, gy - vy, top) + _risk(danger, vx, vy) / 1000
            if score < best_score:
                best_depth, best_score = depth, score
        return best_depth

    @staticmethod
    def _track(
        goal: Point, days: int, velocities: tuple[Point, ...], danger: Point | None
    ) -> Depth:
        """Reach goal in `days` days: hold the needed average velocity.

        Each day take the current that keeps the rest of the trip on that average.
        With a countdown the leftover error feeds into the next day's choice.
        """
        gx, gy = goal
        wx, wy = _mix_closest((gx / days, gy / days), velocities)
        rest_x, rest_y = gx - (days - 1) * wx, gy - (days - 1) * wy
        best_depth, best_score = Depth.DEEP, inf
        for depth, (vx, vy) in zip(_DEPTHS, velocities):
            score = (rest_x - vx) ** 2 + (rest_y - vy) ** 2 + _risk(danger, vx, vy)
            if score < best_score:
                best_depth, best_score = depth, score
        return best_depth

    @staticmethod
    def _station(center: Point, danger: Point | None, drift: Point | None) -> Point:
        """Where to sit (or which line to ride) so the patch stays clear of danger."""
        if danger is None:
            return center
        dx, dy = _wrap(center[0] - danger[0]), _wrap(center[1] - danger[1])
        if drift is None:
            # Holding still: aim for the half of the patch facing away from danger.
            separation = hypot(dx, dy)
            clearance = 1.25 * _R - separation
            if separation == 0 or clearance <= 0:
                return center
            shift = min(clearance, 0.55 * _R) / separation
            return center[0] + dx * shift, center[1] + dy * shift
        # Drifting: slide the ride line sideways until it clears the danger disk.
        speed = hypot(*drift)
        nx, ny = -drift[1] / speed, drift[0] / speed
        lateral = dx * nx + dy * ny
        need = 1.1 * _R - abs(lateral)
        if need <= 0:
            return center
        shift = min(need, 0.6 * _R) * (1 if lateral >= 0 else -1)
        return center[0] + nx * shift, center[1] + ny * shift

    @staticmethod
    def _detour(goal: Point, danger: Point | None) -> Point:
        """If the direct route cuts across danger, pass on the nearer side."""
        if danger is None:
            return goal
        gx, gy = goal
        length_sq = gx * gx + gy * gy
        if length_sq == 0:
            return goal
        # Use the image of the hazard nearest the middle of this wrapped route.
        hx = gx / 2 + _wrap(danger[0] - gx / 2)
        hy = gy / 2 + _wrap(danger[1] - gy / 2)
        along = (hx * gx + hy * gy) / length_sq
        if not 0 < along < 1:
            return goal
        if hypot(hx - along * gx, hy - along * gy) >= 1.1 * _R:
            return goal

        # Two waypoints on opposite sides of the disk, perpendicular to its
        # bearing to the destination. Continue toward the nearer waypoint.
        dx, dy = gx - hx, gy - hy
        length = hypot(dx, dy)
        if length == 0:
            return goal
        offset = 1.35 * _R / length
        left = (hx - dy * offset, hy + dx * offset)
        right = (hx + dy * offset, hy - dx * offset)
        return min((left, right), key=lambda p: p[0] * p[0] + p[1] * p[1])

    def _update_map(self, s: JellyfishSnapshot, target: int) -> None:
        # A baby starts where its parent spawned, before the parent's drift.
        first_turn = s.age_days == 0 or (
            s.age_days == 1 and (s.birth_year, s.birth_day_of_year) != (1, 1)
        )
        vx, vy = (0.0, 0.0) if first_turn else s.velocity
        # We crossed somewhere along yesterday's ride; take its midpoint.
        px, py = -vx / 2, -vy / 2

        for site, inside in (
            (_FOOD, s.in_feeding_zone),
            (_NURSERY, s.in_breeding_zone),
            (_DANGER, s.in_predator_zone),
        ):
            crossed = not first_turn and inside != self._was_inside(s.memory, site)
            self._set_inside(s.memory, site, inside)
            old = self._read_site(s.memory, site)
            if old is None:
                if inside:
                    x = y = 0.0
                    if crossed:
                        # The center lies one radius from the crossing, somewhere
                        # ahead. For danger put it a full radius ahead, so the rim
                        # we just crossed stays right behind us when escaping.
                        ahead = _R if site == _DANGER else _ENTRY_DEPTH
                        speed = hypot(vx, vy)
                        x = px + vx / speed * ahead
                        y = py + vy / speed * ahead
                    self._write_site(s.memory, site, x, y)
                continue
            if first_turn:
                continue  # The parent already recorded this observation at the birth site.
            x, y = _wrap(old[0] - vx), _wrap(old[1] - vy)
            heading_there = site == target and not inside and not crossed
            if heading_there and hypot(x, y) <= _ARRIVED:
                # Well inside the estimated patch yet outside it: the estimate is
                # wrong, so forget it and search again. Off-season sites are kept.
                self._clear_site(s.memory, site)
                continue
            if crossed:
                x, y = _pin_to_rim(x, y, px, py)
            elif not heading_there:
                # Pushing the center we are heading for away from us would just
                # start a chase, so only other sites get nudged.
                x, y = _nudge(x, y, inside)
            self._write_site(s.memory, site, x, y)

    @staticmethod
    def _escape(headings: tuple[int, ...], danger: Point | None) -> Depth:
        """Take the current that leaves the danger disk soonest in the worst case.

        We know we are inside, so the center is within one radius. The estimate may
        be off by tens of degrees, so score each current against the estimate and
        against two copies swung 40 degrees either way around the nearest rim
        point. Backing out the way we came in does well under all three.
        """
        if danger is None:
            return Depth.SURFACE
        cx, cy = _wrap(danger[0]), _wrap(danger[1])
        distance = hypot(cx, cy)
        if distance == 0:
            return Depth.SURFACE
        ux, uy = cx / distance, cy / distance
        distance = min(distance, _R - _ESCAPE_MARGIN)
        rim_x, rim_y = ux * (distance - _R), uy * (distance - _R)
        centers = [
            (
                rim_x + _R * (ux * cos(a) - uy * sin(a)),
                rim_y + _R * (ux * sin(a) + uy * cos(a)),
            )
            for a in (0.0, _ESCAPE_SWING, -_ESCAPE_SWING)
        ]
        best_depth, best_days = Depth.SURFACE, inf
        for depth in _DEPTHS:
            if headings[depth]:
                vx, vy = _velocity(depth, headings[depth])
                days = max(_days_inside(vx, vy, x, y) for x, y in centers)
                if days < best_days:
                    best_depth, best_days = depth, days
        return best_depth

    def _simple(self, s: JellyfishSnapshot, breeding: bool, inside: bool) -> Action:
        """In small-memory games remember the current that last entered the target."""
        if s.in_predator_zone:
            return Action(Depth.SURFACE)
        if inside:
            if s.depth != Depth.DEEP:
                s.memory[0] = (breeding << 2) | s.depth
            return Action(Depth.DEEP)
        if s.depth == Depth.DEEP:
            last_target, entry_depth = divmod(s.memory[0], 4)
            if last_target == breeding:
                return Action(Depth(entry_depth))
        return self._explore(s)

    def _explore(self, s: JellyfishSnapshot) -> Action:
        # Ride one current for a while, so the search covers new water.
        if s.depth == Depth.DEEP or self.rng.randrange(12) == 0:
            return Action(self.rng.choice(_SEARCH_DEPTHS))
        return Action(s.depth)
