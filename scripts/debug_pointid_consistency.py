#!/usr/bin/env python
"""
Debug script to check pointId consistency between make_candidate and get_current_point_id
"""

import sys
import os
import math

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set up MatterSim
sys.path.append("buildpy36")
sys.path.append("Matterport_Simulator/build/")

try:
    import MatterSim
except ImportError:
    print("Error: MatterSim not found")
    sys.exit(1)


def get_current_point_id(state, num_horizontal_views):
    """Same as in agent.py"""
    angle_increment = 360.0 / num_horizontal_views
    angle_increment_rad = math.radians(angle_increment)

    elevation = state.elevation

    if elevation < -0.2:
        elev_level = 0
    elif elevation > 0.2:
        elev_level = 2
    else:
        elev_level = 1

    heading_normalized = state.heading % (2 * math.pi)
    horiz_idx_raw = heading_normalized / angle_increment_rad
    horiz_idx = int(round(horiz_idx_raw)) % num_horizontal_views

    point_id = elev_level * num_horizontal_views + horiz_idx
    return point_id


def test_pointid_consistency():
    """Test if pointId calculated from state matches ix in make_candidate"""
    print("=" * 80)
    print("Testing pointId consistency between make_candidate and get_current_point_id")
    print("=" * 80)

    scanId = "ZMojNkEp431"
    viewpointId = "2f4d90acd4024c269fb0efe49a8ac540"
    num_horizontal_views = 8
    num_total_views = 3 * num_horizontal_views
    angle_increment_rad = math.radians(360.0 / num_horizontal_views)

    sim = MatterSim.Simulator()
    sim.setRenderingEnabled(False)
    sim.setDiscretizedViewingAngles(False)
    sim.setCameraResolution(640, 480)
    sim.setCameraVFOV(math.radians(60))
    sim.initialize()

    mismatches = []
    matches = 0

    print(f"\nTesting with {num_horizontal_views} views (45° increments)")
    print(f"Checking first {min(16, num_total_views)} views:\n")

    for ix in range(min(16, num_total_views)):
        if ix == 0:
            sim.newEpisode([scanId], [viewpointId], [0], [math.radians(-30)])
        elif ix % num_horizontal_views == 0:
            sim.makeAction([0], [0.0], [math.radians(30)])
        else:
            sim.makeAction([0], [angle_increment_rad], [0])

        state = sim.getState()[0]
        calculated_point_id = get_current_point_id(state, num_horizontal_views)
        expected_point_id = ix

        if calculated_point_id == expected_point_id:
            matches += 1
            match_str = "✓"
        else:
            mismatches.append({
                'ix': ix,
                'expected': expected_point_id,
                'calculated': calculated_point_id,
                'heading': state.heading,
                'elevation': state.elevation
            })
            match_str = "✗"

        print(f"  ix={ix:2d}: expected={expected_point_id:2d}, calc={calculated_point_id:2d} {match_str} "
              f"(heading={math.degrees(state.heading):6.2f}°, elev={math.degrees(state.elevation):6.2f}°)")

    print(f"\nMatches: {matches}/{min(16, num_total_views)}")
    if mismatches:
        print(f"Mismatches: {len(mismatches)}")
        print("\nDetailed mismatch information:")
        for m in mismatches:
            print(f"  ix={m['ix']}: expected={m['expected']}, calculated={m['calculated']}")
            print(f"    heading={math.degrees(m['heading']):.2f}°, elevation={math.degrees(m['elevation']):.2f}°")
            # Calculate what horiz_idx should be
            elev_level = m['expected'] // num_horizontal_views
            expected_horiz_idx = m['expected'] % num_horizontal_views
            heading_normalized = m['heading'] % (2 * math.pi)
            calc_horiz_idx_raw = heading_normalized / angle_increment_rad
            calc_horiz_idx = int(round(calc_horiz_idx_raw)) % num_horizontal_views
            print(f"    expected_horiz_idx={expected_horiz_idx}, calc_horiz_idx={calc_horiz_idx} "
                  f"(raw={calc_horiz_idx_raw:.2f})")
    else:
        print("✓ All pointIds match!")


if __name__ == "__main__":
    test_pointid_consistency()

