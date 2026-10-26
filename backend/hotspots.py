"""Group reports that sit within a radius of each other."""

from math import asin, cos, radians, sin, sqrt

EARTH_RADIUS_M = 6_371_000


def haversine_m(lat1, lng1, lat2, lng2):
    dlat, dlng = radians(lat2 - lat1), radians(lng2 - lng1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlng / 2) ** 2
    return 2 * EARTH_RADIUS_M * asin(sqrt(a))


def find_hotspots(reports, radius_m=200):
    """Chain reports into groups (A near B, B near C -> one group). Returns groups of 2+."""
    # ponytail: O(n^2) pair scan, fine for thousands of reports; use a geohash grid beyond that.
    unvisited = list(range(len(reports)))
    hotspots = []
    while unvisited:
        group, queue = [], [unvisited.pop()]
        while queue:
            i = queue.pop()
            group.append(i)
            near = [j for j in unvisited if haversine_m(
                reports[i]["lat"], reports[i]["lng"], reports[j]["lat"], reports[j]["lng"]) <= radius_m]
            for j in near:
                unvisited.remove(j)
            queue.extend(near)
        if len(group) >= 2:
            members = [reports[i] for i in group]
            hotspots.append({
                "lat": sum(r["lat"] for r in members) / len(members),
                "lng": sum(r["lng"] for r in members) / len(members),
                "count": len(members),
                "report_ids": [r["id"] for r in members],
            })
    return sorted(hotspots, key=lambda h: -h["count"])


if __name__ == "__main__":
    # One degree of latitude is ~111.2 km.
    assert abs(haversine_m(28.0, 77.0, 29.0, 77.0) - 111_195) < 50
    pts = [
        {"id": "a", "lat": 28.4600, "lng": 77.4920},
        {"id": "b", "lat": 28.4615, "lng": 77.4920},  # ~167 m from a
        {"id": "c", "lat": 28.4630, "lng": 77.4920},  # ~167 m from b, ~334 m from a: chained
        {"id": "d", "lat": 28.5000, "lng": 77.4920},  # far away
    ]
    hs = find_hotspots(pts)
    assert len(hs) == 1 and sorted(hs[0]["report_ids"]) == ["a", "b", "c"], hs
    print("hotspots self-check ok")
