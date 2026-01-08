#!/usr/bin/env python
"""
Test script to verify the correctness of action and candidate logic
in continuous sim mode (non-discretized) compared to discrete sim mode (30-degree increments).

This script tests:
1. makeAction behavior in discrete vs non-discrete modes
2. make_candidate pointId calculation in both modes
3. make_equiv_action behavior in both modes
4. Consistency between modes
"""

import sys
import os
import math
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Mock args for testing
class MockArgs:
    def __init__(self, panoramic_horizontal_views=12, vfov=60):
        self.panoramic_horizontal_views = panoramic_horizontal_views
        self.vfov = vfov
        self.test_only = False


# Set up environment
try:
    sys.path.append("buildpy36")
    sys.path.append("Matterport_Simulator/build/")
    import MatterSim
except ImportError:
    print("Warning: MatterSim not found. Some tests may fail.")
    MatterSim = None

try:
    import r2r_src.vln_utils as vln_utils
except ImportError:
    print("Warning: vln_utils not found. Some tests may fail.")
    vln_utils = None


def test_make_action_discrete_vs_continuous():
    """Test makeAction behavior in discrete vs continuous modes"""
    print("=" * 80)
    print("Testing makeAction behavior")
    print("=" * 80)

    # Test discrete mode (12 views, 30-degree increments)
    print("\n1. Testing DISCRETE mode (12 views, 30-degree increments)")
    sim_discrete = MatterSim.Simulator()
    sim_discrete.setRenderingEnabled(False)
    sim_discrete.setDiscretizedViewingAngles(True)
    sim_discrete.setCameraResolution(640, 480)
    sim_discrete.setCameraVFOV(math.radians(60))
    sim_discrete.initialize()

    # Initialize episode
    scanId = "ZMojNkEp431"
    viewpointId = "2f4d90acd4024c269fb0efe49a8ac540"
    sim_discrete.newEpisode([scanId], [viewpointId], [0], [math.radians(-30)])

    discrete_states = []
    for i in range(3):  # Test 3 horizontal rotations
        state = sim_discrete.getState()[0]
        discrete_states.append(
            {
                "viewIndex": state.viewIndex,
                "heading": state.heading,
                "elevation": state.elevation,
            }
        )
        print(
            f"  Step {i}: viewIndex={state.viewIndex}, heading={math.degrees(state.heading):.2f}°, elevation={math.degrees(state.elevation):.2f}°"
        )
        if i < 2:
            sim_discrete.makeAction([0], [1.0], [0])  # Rotate right

    # Test continuous mode (8 views, 45-degree increments)
    print("\n2. Testing CONTINUOUS mode (8 views, 45-degree increments)")
    sim_continuous = MatterSim.Simulator()
    sim_continuous.setRenderingEnabled(False)
    sim_continuous.setDiscretizedViewingAngles(False)
    sim_continuous.setCameraResolution(640, 480)
    sim_continuous.setCameraVFOV(math.radians(60))
    sim_continuous.initialize()

    # Initialize episode
    sim_continuous.newEpisode([scanId], [viewpointId], [0], [math.radians(-30)])

    angle_increment_rad = math.radians(360.0 / 8)  # 45 degrees
    continuous_states = []
    for i in range(3):  # Test 3 horizontal rotations
        state = sim_continuous.getState()[0]
        continuous_states.append(
            {
                "viewIndex": state.viewIndex,
                "heading": state.heading,
                "elevation": state.elevation,
            }
        )
        print(
            f"  Step {i}: viewIndex={state.viewIndex}, heading={math.degrees(state.heading):.2f}°, elevation={math.degrees(state.elevation):.2f}°"
        )
        if i < 2:
            sim_continuous.makeAction(
                [0], [angle_increment_rad], [0]
            )  # Rotate right by 45 degrees

    # Compare heading changes
    print("\n3. Comparing heading changes:")
    print("  Discrete mode heading increments:")
    for i in range(1, len(discrete_states)):
        heading_diff = discrete_states[i]["heading"] - discrete_states[i - 1]["heading"]
        # Normalize to [0, 2π)
        heading_diff = (heading_diff + math.pi) % (2 * math.pi) - math.pi
        print(f"    Step {i-1} -> {i}: {math.degrees(heading_diff):.2f}°")

    print("  Continuous mode heading increments:")
    for i in range(1, len(continuous_states)):
        heading_diff = (
            continuous_states[i]["heading"] - continuous_states[i - 1]["heading"]
        )
        # Normalize to [0, 2π)
        heading_diff = (heading_diff + math.pi) % (2 * math.pi) - math.pi
        print(
            f"    Step {i-1} -> {i}: {math.degrees(heading_diff):.2f}° (expected: {math.degrees(angle_increment_rad):.2f}°)"
        )

    return discrete_states, continuous_states


def test_make_candidate_pointid_calculation():
    """Test pointId calculation in make_candidate for both modes"""
    print("\n" + "=" * 80)
    print("Testing make_candidate pointId calculation")
    print("=" * 80)

    # We need to import env module, but we need to mock args first
    # For now, let's test the logic directly

    scanId = "ZMojNkEp431"
    viewpointId = "2f4d90acd4024c269fb0efe49a8ac540"

    # Test discrete mode (12 views)
    print("\n1. Testing DISCRETE mode (12 views)")
    num_horizontal_views = 12
    num_total_views = 3 * num_horizontal_views
    angle_increment = 360.0 / num_horizontal_views
    angle_increment_rad = math.radians(angle_increment)

    sim = MatterSim.Simulator()
    sim.setRenderingEnabled(False)
    sim.setDiscretizedViewingAngles(True)
    sim.setCameraResolution(640, 480)
    sim.setCameraVFOV(math.radians(60))
    sim.initialize()

    discrete_point_ids = []
    for ix in range(min(24, num_total_views)):  # Test first 24 views
        if ix == 0:
            sim.newEpisode([scanId], [viewpointId], [0], [math.radians(-30)])
        elif ix % num_horizontal_views == 0:
            sim.makeAction([0], [1.0], [1.0])
        else:
            sim.makeAction([0], [1.0], [0])

        state = sim.getState()[0]
        elev_level = ix // num_horizontal_views
        horiz_idx = ix % num_horizontal_views
        expected_point_id = elev_level * num_horizontal_views + horiz_idx

        discrete_point_ids.append(
            {
                "ix": ix,
                "viewIndex": state.viewIndex,
                "pointId": expected_point_id,
                "elevation": state.elevation,
                "heading": state.heading,
            }
        )

        if ix < 5 or (ix % num_horizontal_views == 0):
            print(
                f"  ix={ix:2d}: viewIndex={state.viewIndex:2d}, pointId={expected_point_id:2d}, "
                f"elev={math.degrees(state.elevation):6.2f}°, heading={math.degrees(state.heading):6.2f}°"
            )

    # Test continuous mode (8 views)
    print("\n2. Testing CONTINUOUS mode (8 views)")
    num_horizontal_views = 8
    num_total_views = 3 * num_horizontal_views
    angle_increment = 360.0 / num_horizontal_views
    angle_increment_rad = math.radians(angle_increment)

    sim = MatterSim.Simulator()
    sim.setRenderingEnabled(False)
    sim.setDiscretizedViewingAngles(False)
    sim.setCameraResolution(640, 480)
    sim.setCameraVFOV(math.radians(60))
    sim.initialize()

    continuous_point_ids = []
    for ix in range(min(24, num_total_views)):  # Test first 24 views
        if ix == 0:
            sim.newEpisode([scanId], [viewpointId], [0], [math.radians(-30)])
        elif ix % num_horizontal_views == 0:
            sim.makeAction([0], [0.0], [math.radians(30)])
        else:
            sim.makeAction([0], [angle_increment_rad], [0])

        state = sim.getState()[0]
        elev_level = ix // num_horizontal_views
        horiz_idx = ix % num_horizontal_views
        expected_point_id = elev_level * num_horizontal_views + horiz_idx

        # Calculate pointId from state (as in get_current_point_id)
        elevation = state.elevation
        if elevation < -0.2:
            calc_elev_level = 0
        elif elevation > 0.2:
            calc_elev_level = 2
        else:
            calc_elev_level = 1

        heading_normalized = state.heading % (2 * math.pi)
        calc_horiz_idx = (
            int(round(heading_normalized / angle_increment_rad)) % num_horizontal_views
        )
        calc_point_id = calc_elev_level * num_horizontal_views + calc_horiz_idx

        continuous_point_ids.append(
            {
                "ix": ix,
                "viewIndex": state.viewIndex,
                "pointId": expected_point_id,
                "calc_pointId": calc_point_id,
                "elevation": state.elevation,
                "heading": state.heading,
                "calc_elev_level": calc_elev_level,
                "calc_horiz_idx": calc_horiz_idx,
            }
        )

        if ix < 5 or (ix % num_horizontal_views == 0):
            match = "✓" if expected_point_id == calc_point_id else "✗"
            print(
                f"  ix={ix:2d}: viewIndex={state.viewIndex:2d}, pointId={expected_point_id:2d}, "
                f"calc_pointId={calc_point_id:2d} {match}, "
                f"elev={math.degrees(state.elevation):6.2f}°({calc_elev_level}), "
                f"heading={math.degrees(state.heading):6.2f}°(idx={calc_horiz_idx})"
            )

    # Check for mismatches
    mismatches = [p for p in continuous_point_ids if p["pointId"] != p["calc_pointId"]]
    if mismatches:
        print(f"\n  ⚠️  Found {len(mismatches)} mismatches in pointId calculation!")
        for m in mismatches[:5]:
            print(
                f"    ix={m['ix']}: expected={m['pointId']}, calculated={m['calc_pointId']}"
            )
    else:
        print("\n  ✓ All pointId calculations match!")

    return discrete_point_ids, continuous_point_ids


def test_get_obs_viewid_calculation():
    """Test viewId calculation in _get_obs for continuous mode"""
    print("\n" + "=" * 80)
    print("Testing _get_obs viewId calculation")
    print("=" * 80)

    scanId = "ZMojNkEp431"
    viewpointId = "2f4d90acd4024c269fb0efe49a8ac540"

    # Test continuous mode (8 views)
    print("\nTesting CONTINUOUS mode (8 views)")
    num_horizontal_views = 8
    angle_increment_rad = math.radians(360.0 / num_horizontal_views)

    sim = MatterSim.Simulator()
    sim.setRenderingEnabled(False)
    sim.setDiscretizedViewingAngles(False)
    sim.setCameraResolution(640, 480)
    sim.setCameraVFOV(math.radians(60))
    sim.initialize()

    # Test different states
    test_cases = [
        (0, 0, math.radians(-30)),  # Initial state
        (1, 0, math.radians(-30)),  # After one right rotation
        (8, 0, 0),  # After moving up one level
        (9, 0, 0),  # After moving up and one right rotation
    ]

    for step, expected_elev_level, expected_elevation in test_cases:
        if step == 0:
            sim.newEpisode([scanId], [viewpointId], [0], [math.radians(-30)])
        elif step == 1:
            sim.newEpisode([scanId], [viewpointId], [0], [math.radians(-30)])
            sim.makeAction([0], [angle_increment_rad], [0])
        elif step == 8:
            sim.newEpisode([scanId], [viewpointId], [0], [math.radians(-30)])
            sim.makeAction([0], [0.0], [math.radians(30)])
        elif step == 9:
            sim.newEpisode([scanId], [viewpointId], [0], [math.radians(-30)])
            sim.makeAction([0], [0.0], [math.radians(30)])
            sim.makeAction([0], [angle_increment_rad], [0])

        state = sim.getState()[0]

        # Calculate viewId as in _get_obs
        if state.elevation < -0.2:
            elev_level = 0
        elif state.elevation > 0.2:
            elev_level = 2
        else:
            elev_level = 1

        heading_normalized = state.heading % (2 * math.pi)
        horiz_idx = (
            int(round(heading_normalized / angle_increment_rad)) % num_horizontal_views
        )
        point_id = elev_level * num_horizontal_views + horiz_idx

        expected_point_id = step

        match = "✓" if point_id == expected_point_id else "✗"
        print(
            f"  Step {step:2d}: viewIndex={state.viewIndex:2d}, "
            f"pointId={point_id:2d} (expected={expected_point_id:2d}) {match}, "
            f"elev={math.degrees(state.elevation):6.2f}°(level={elev_level}), "
            f"heading={math.degrees(state.heading):6.2f}°(idx={horiz_idx})"
        )


def test_make_equiv_action_consistency():
    """Test make_equiv_action consistency between discrete and continuous modes"""
    print("\n" + "=" * 80)
    print("Testing make_equiv_action consistency")
    print("=" * 80)

    scanId = "ZMojNkEp431"
    viewpointId = "2f4d90acd4024c269fb0efe49a8ac540"

    # Test case: navigate from pointId 0 to pointId 9
    print("\nTest case: Navigate from pointId 0 to pointId 9")

    # Discrete mode (12 views)
    print("\n1. DISCRETE mode (12 views):")
    print("   pointId 0 -> 9: elev_level 0->0, horiz_idx 0->9")
    print("   Expected: 0 up actions, 9 right actions")

    # Continuous mode (8 views)
    print("\n2. CONTINUOUS mode (8 views):")
    print("   pointId 0 -> 9: elev_level 0->1, horiz_idx 0->1")
    print("   Expected: 1 up action, 1 right action")

    num_horizontal_views = 8
    src_point = 0
    trg_point = 9

    src_level = src_point // num_horizontal_views
    trg_level = trg_point // num_horizontal_views
    src_horiz = src_point % num_horizontal_views
    trg_horiz = trg_point % num_horizontal_views

    print(f"\n   src_point={src_point}: level={src_level}, horiz={src_horiz}")
    print(f"   trg_point={trg_point}: level={trg_level}, horiz={trg_horiz}")
    print(f"   Expected up actions: {max(0, trg_level - src_level)}")
    print(
        f"   Expected right actions: {trg_horiz - src_horiz if trg_level == src_level else 'N/A (need to calculate from heading)'}"
    )


def test_base_heading_calculation():
    """Test base_heading calculation in make_candidate"""
    print("\n" + "=" * 80)
    print("Testing base_heading calculation")
    print("=" * 80)

    # Test discrete mode
    print("\n1. DISCRETE mode (12 views):")
    num_horizontal_views = 12
    angle_increment_rad = math.radians(360.0 / num_horizontal_views)

    for viewId in [0, 5, 11, 12, 24]:
        base_heading = (viewId % num_horizontal_views) * angle_increment_rad
        print(
            f"   viewId={viewId:2d}: base_heading={math.degrees(base_heading):6.2f}° "
            f"(horiz_idx={viewId % num_horizontal_views})"
        )

    # Test continuous mode
    print("\n2. CONTINUOUS mode (8 views):")
    num_horizontal_views = 8
    angle_increment_rad = math.radians(360.0 / num_horizontal_views)

    for viewId in [0, 5, 7, 8, 16]:
        base_heading = (viewId % num_horizontal_views) * angle_increment_rad
        print(
            f"   viewId={viewId:2d}: base_heading={math.degrees(base_heading):6.2f}° "
            f"(horiz_idx={viewId % num_horizontal_views})"
        )


def test_env_make_candidate_with_mock():
    """Test make_candidate function with mocked args"""
    print("\n" + "=" * 80)
    print("Testing env.make_candidate with different panoramic_horizontal_views")
    print("=" * 80)

    if MatterSim is None:
        print("  ⚠️  Skipping: MatterSim not available")
        return

    # We need to test this by actually importing and using the env module
    # But we need to set args.panoramic_horizontal_views first
    print(
        "\nNote: This test requires running with different args.panoramic_horizontal_views"
    )
    print("      values. Run the script with:")
    print(
        "        python -c 'import sys; sys.path.insert(0, \"scripts\"); from test_continuous_sim import *; test_env_make_candidate_with_mock()'"
    )
    print("      after setting args.panoramic_horizontal_views")


def test_agent_make_equiv_action_with_mock():
    """Test make_equiv_action function with mocked args"""
    print("\n" + "=" * 80)
    print("Testing agent.make_equiv_action with different panoramic_horizontal_views")
    print("=" * 80)

    if MatterSim is None:
        print("  ⚠️  Skipping: MatterSim not available")
        return

    print(
        "\nNote: This test requires running with different args.panoramic_horizontal_views"
    )
    print("      values. Run the script with:")
    print(
        "        python -c 'import sys; sys.path.insert(0, \"scripts\"); from test_continuous_sim import *; test_agent_make_equiv_action_with_mock()'"
    )
    print("      after setting args.panoramic_horizontal_views")


def test_heading_elevation_to_pointid():
    """Test conversion from heading/elevation to pointId"""
    print("\n" + "=" * 80)
    print("Testing heading/elevation to pointId conversion")
    print("=" * 80)

    num_horizontal_views = 8
    angle_increment_rad = math.radians(360.0 / num_horizontal_views)

    test_cases = [
        # (heading_deg, elevation_deg, base_view_index, expected_point_id)
        (0, -30, 0, 0),  # Initial state
        (45, -30, 0, 1),  # One rotation right
        (90, -30, 0, 2),  # Two rotations right
        (0, 0, 0, 8),  # After moving up
        (45, 0, 0, 9),  # After moving up and one rotation
    ]

    print(
        f"\nTesting with num_horizontal_views={num_horizontal_views} (45° increments)"
    )

    for heading_deg, elevation_deg, base_view_index, expected_point_id in test_cases:
        heading = math.radians(heading_deg)
        elevation = math.radians(elevation_deg)

        # Calculate base_heading
        base_heading = (base_view_index % num_horizontal_views) * angle_increment_rad

        # Calculate relative heading
        relative_heading = (heading - base_heading) % (2 * math.pi)

        # Determine elevation level
        if elevation < -0.2:
            elev_level = 0
        elif elevation > 0.2:
            elev_level = 2
        else:
            elev_level = 1

        # Calculate horizontal index
        horiz_idx = (
            int(round(relative_heading / angle_increment_rad)) % num_horizontal_views
        )

        # Calculate pointId
        point_id = elev_level * num_horizontal_views + horiz_idx

        match = "✓" if point_id == expected_point_id else "✗"
        print(
            f"  heading={heading_deg:5.1f}°, elev={elevation_deg:5.1f}°, "
            f"base_idx={base_view_index:2d} -> pointId={point_id:2d} "
            f"(expected={expected_point_id:2d}) {match}"
        )
        if point_id != expected_point_id:
            print(
                f"    Details: elev_level={elev_level}, horiz_idx={horiz_idx}, "
                f"relative_heading={math.degrees(relative_heading):.2f}°"
            )


def main():
    """Run all tests"""
    print("=" * 80)
    print("Testing Continuous Sim Mode vs Discrete Sim Mode")
    print("=" * 80)

    if MatterSim is None:
        print("\n⚠️  Warning: MatterSim not available. Some tests will be skipped.")
        print("   Please ensure MatterSim is properly installed.")

    try:
        # Test 1: makeAction behavior
        if MatterSim is not None:
            discrete_states, continuous_states = (
                test_make_action_discrete_vs_continuous()
            )

        # Test 2: pointId calculation
        if MatterSim is not None:
            discrete_point_ids, continuous_point_ids = (
                test_make_candidate_pointid_calculation()
            )

        # Test 3: _get_obs viewId calculation
        if MatterSim is not None:
            test_get_obs_viewid_calculation()

        # Test 4: make_equiv_action consistency
        test_make_equiv_action_consistency()

        # Test 5: base_heading calculation
        test_base_heading_calculation()

        # Test 6: heading/elevation to pointId conversion
        test_heading_elevation_to_pointid()

        # Test 7: env.make_candidate (requires actual env module)
        test_env_make_candidate_with_mock()

        # Test 8: agent.make_equiv_action (requires actual agent module)
        test_agent_make_equiv_action_with_mock()

        print("\n" + "=" * 80)
        print("All tests completed!")
        print("=" * 80)
        print("\nNext steps:")
        print("1. Run this script to verify basic logic")
        print(
            "2. Test with actual env.py and agent.py by setting args.panoramic_horizontal_views"
        )
        print(
            "3. Compare results between discrete (12 views) and continuous (8 views) modes"
        )

    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
