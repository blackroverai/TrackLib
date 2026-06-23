"""Category-group association gate (BlackRover fork) — name-keyed.

`role -> {name -> group_str}` config is seeded on the tracker at init; per role
the tracker composes a `{class_id -> group_str}` submap from that role's name map
and the detector's runtime `names` (compose_group_map). matching.py gates on the
string category_group: a label that flaps within one group (car<->truck<->bus all
"vehicle") still associates, while different groups (person vs vehicle) stay apart.
An unmapped class falls through to its own class name; an empty map falls through
to the raw category (byte-identical to upstream).
"""

import numpy as np

from tracker.trackers import matching
from tracker.trackers.tracklet import compose_group_map, _resolve_group


# name->group config (per role), and per-role runtime names (id->name). The two
# roles deliberately reuse the SAME ids for DIFFERENT classes to prove isolation:
#   ground: id 2 = private_car (vehicle), id 7 = truck (vehicle), id 0 = person
#   aerial: id 2 = person,                id 7 = private_car (vehicle)
GROUND_CFG = {"private_car": "vehicle", "truck": "vehicle", "person": "person"}
AERIAL_CFG = {"private_car": "vehicle", "person": "person"}
GROUND_NAMES = {2: "private_car", 7: "truck", 0: "person"}
AERIAL_NAMES = {2: "person", 7: "private_car"}


class _MockTrack:
    """Minimal stand-in: iou_distance reads `.tlbr` and `.category_group`."""
    def __init__(self, tlbr, category, group_map=None):
        self.tlbr = np.asarray(tlbr, dtype=float)
        self.category = category
        self.category_group = _resolve_group(category, group_map)


# --- compose_group_map: (name->group) composed with runtime names -> id->group --

def test_compose_maps_ids_to_group_strings():
    assert compose_group_map(GROUND_CFG, GROUND_NAMES) == {2: "vehicle", 7: "vehicle", 0: "person"}


def test_compose_unmapped_class_falls_through_to_its_name():
    gm = compose_group_map({"truck": "vehicle"}, {7: "truck", 9: "traffic_light"})
    assert gm[7] == "vehicle"
    assert gm[9] == "traffic_light"             # unmapped -> own class name (gates as itself)


def test_compose_empty_inputs_are_identity():
    assert compose_group_map({}, {2: "car"}) == {}        # no config
    assert compose_group_map({"car": "vehicle"}, {}) == {}  # no names


def test_role_isolation_same_id_different_group():
    g = compose_group_map(GROUND_CFG, GROUND_NAMES)
    a = compose_group_map(AERIAL_CFG, AERIAL_NAMES)
    assert g[2] == "vehicle"     # id 2 is a vehicle under ground
    assert a[2] == "person"      # ...but a person under aerial


# --- _resolve_group: id -> group under a composed submap ----------------------

def test_resolve_group_string_or_raw_category():
    gm = compose_group_map(GROUND_CFG, GROUND_NAMES)
    assert _resolve_group(2, gm) == "vehicle"
    assert _resolve_group(0, gm) == "person"
    assert _resolve_group(5, gm) == 5        # id absent from a non-empty map -> raw category
    assert _resolve_group(2, {}) == 2        # empty map -> identity (byte-identical upstream)
    assert _resolve_group(2, None) == 2


# --- the association gate (matching.iou_distance) -----------------------------

def test_iou_distance_gates_on_group_not_raw_class():
    gm = compose_group_map(GROUND_CFG, GROUND_NAMES)
    track = [_MockTrack([0, 0, 10, 10], category=7, group_map=gm)]   # truck (vehicle)
    dets = [
        _MockTrack([0, 0, 10, 10], category=2, group_map=gm),        # car: same vehicle group
        _MockTrack([0, 0, 10, 10], category=0, group_map=gm),        # person: different group
    ]
    cost = matching.iou_distance(track, dets)
    assert np.isfinite(cost[0, 0]), "same-group flap (truck<->car) must remain associable"
    assert not np.isfinite(cost[0, 1]), "cross-group (vehicle vs person) must be gated to inf"


def test_no_map_falls_through_to_raw_category():
    track = [_MockTrack([0, 0, 10, 10], category=7)]    # truck, no group map
    dets = [
        _MockTrack([0, 0, 10, 10], category=7),         # truck: same raw class
        _MockTrack([0, 0, 10, 10], category=2),         # car: different raw class
    ]
    cost = matching.iou_distance(track, dets)
    assert np.isfinite(cost[0, 0]), "same raw class associates"
    assert not np.isfinite(cost[0, 1]), "different raw class gated (upstream behaviour)"
