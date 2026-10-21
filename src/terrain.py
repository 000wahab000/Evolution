"""Ground courses as data. CreatureEnv builds all of its ground from a TerrainSpec.

Coordinates are pymunk's (y grows downward); a segment's top surface is y - radius.
"""

from dataclasses import dataclass, replace

import pymunk

TERRAIN_VERSION = 1
SPAWN_FLAT_END = 600.0  # torso spawns at x=400, so 200 px of flat ground ahead of it
GROUND_Y = 500.0


@dataclass(frozen=True)
class Segment:
    a: tuple
    b: tuple
    radius: float
    friction: float


@dataclass(frozen=True)
class TerrainSpec:
    segments: tuple  # added to the space in this order
    name: str = "custom"
    legacy: bool = False  # legacy: own static body per extra segment, old fallen check

    def scaled_friction(self, scale):
        if scale == 1.0:
            return self
        return replace(self, segments=tuple(replace(s, friction=s.friction * scale)
                                            for s in self.segments))

    def ground_top(self, x):
        """Highest surface point (smallest y) of any segment covering x, or None off the course."""
        tops = []
        for s in self.segments:
            (ax, ay), (bx, by) = sorted((s.a, s.b))
            if ax <= x <= bx:
                y = ay if bx == ax else ay + (by - ay) * (x - ax) / (bx - ax)
                tops.append(y - s.radius)
        return min(tops) if tops else None

    def polyline(self):
        """Course points left to right, for courses built as one continuous line."""
        return [list(self.segments[0].a)] + [list(s.b) for s in self.segments]


def legacy_course():
    """Today's ground, bug included: the 'downhill' reuses the uphill formula (40 px cliff at x=1600)."""
    return TerrainSpec(segments=(
        Segment((-10000, 500), (3000, 500), 5.0, 1.2),
        Segment((600, 500), (1600, 460.0), 4, 1.8),
        Segment((1600, 500), (2600, 460.0), 4, 1.8),
    ), name="legacy", legacy=True)


def _course(points, name, flat_friction=1.2, slope_friction=1.8, radius=5.0):
    segs = tuple(Segment(tuple(a), tuple(b), radius, flat_friction if a[1] == b[1] else slope_friction)
                 for a, b in zip(points, points[1:]))
    return TerrainSpec(segments=segs, name=name)


def fixed_course():
    """The intended course: flat, uphill to 1600, real downhill back to y=500 at 2600, flat to 3000."""
    return _course([(-10000, 500), (600, 500), (1600, 460), (2600, 500), (3000, 500)], "fixed")


def random_course(rng, max_slope, length=3000, n_segments=10, min_slope=0.0, name="random"):
    """Continuous piecewise-linear course: flat to x=600, then n segments whose slope magnitude
    is U[min_slope, max_slope] with a random up/down sign."""
    xs = [SPAWN_FLAT_END + (length - SPAWN_FLAT_END) * i / n_segments for i in range(n_segments + 1)]
    points = [(-10000.0, GROUND_Y), (xs[0], GROUND_Y)]
    for x0, x1 in zip(xs, xs[1:]):
        slope = rng.uniform(min_slope, max_slope) * (1 if rng.random() < 0.5 else -1)
        points.append((x1, points[-1][1] - slope * (x1 - x0)))  # y down: positive slope = uphill
    return _course(points, name, flat_friction=1.2, slope_friction=1.2)


def add_terrain(space, spec):
    """Add every segment of spec to space and return the shapes."""
    shapes = []
    for i, s in enumerate(spec.segments):
        if spec.legacy and i > 0:  # matches the old terrain_changing(): one static body each
            body = pymunk.Body(body_type=pymunk.Body.STATIC)
            shape = pymunk.Segment(body, s.a, s.b, s.radius)
            shape.friction = s.friction
            space.add(body, shape)
        else:
            shape = pymunk.Segment(space.static_body, s.a, s.b, s.radius)
            shape.friction = s.friction
            space.add(shape)
        shapes.append(shape)
    return shapes
