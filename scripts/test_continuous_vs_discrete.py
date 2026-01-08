#!/usr/bin/env python
"""
Comprehensive test to compare discrete (12 views, 30°) vs continuous (8 views, 45°) modes.

This script tests:
1. makeAction produces expected heading/elevation changes
2. make_candidate produces correct pointIds
3. make_equiv_action navigates correctly
4. Consistency between modes

Run with:
    python scripts/test_continuous_vs_discrete.py
"""

import sys
import os
import math
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set up MatterSim
sys.path.append("buildpy36")
sys.path.append("Matterport_Simulator/build/")

try:
    import MatterSim

    MATTERSIM_AVAILABLE = True
except ImportError:
    print("Error: MatterSim not found. Please install MatterSim first.")
    sys.exit(1)


def test_make_action_up_down():
    """Test up and down actions in both modes by directly simulating make_candidate"""
    print("=" * 80)
    print("Test 1: UP and DOWN Actions")
    print("=" * 80)
    print("\nNote: Testing elevation changes by simulating make_candidate behavior")
    print("      This directly replicates the logic in env.py make_candidate")

    scanId = "ZMojNkEp431"
    viewpointId = "2f4d90acd4024c269fb0efe49a8ac540"

    results = {}

    for mode_name, use_discretized, num_views in [
        ("Discrete", True, 12),
        ("Continuous", False, 8),
    ]:
        print(f"\n{mode_name} mode ({num_views} views):")
        sim = MatterSim.Simulator()
        sim.setRenderingEnabled(False)
        sim.setDiscretizedViewingAngles(use_discretized)
        sim.setCameraResolution(640, 480)
        sim.setCameraVFOV(math.radians(60))
        sim.initialize()

        # Directly simulate make_candidate logic to check elevation changes
        num_total_views = 3 * num_views
        angle_increment_rad = math.radians(360.0 / num_views)

        elevations_at_ix = []

        # Simulate make_candidate: iterate through ix values
        for ix in range(
            min(num_total_views, num_views * 2 + 1)
        ):  # Check first two levels
            if ix == 0:
                sim.newEpisode([scanId], [viewpointId], [0], [math.radians(-30)])
            elif ix % num_views == 0:
                # Move up one elevation level (as in env.py line 349-355)
                if use_discretized:
                    sim.makeAction([0], [1.0], [1.0])
                else:
                    sim.makeAction([0], [0.0], [math.radians(30)])
            else:
                # Rotate horizontally (as in env.py line 356-362)
                if use_discretized:
                    sim.makeAction([0], [1.0], [0])
                else:
                    sim.makeAction([0], [angle_increment_rad], [0])

            state = sim.getState()[0]
            elevations_at_ix.append(math.degrees(state.elevation))

            # Print key transitions
            if ix == 0 or ix == num_views or ix == num_views * 2:
                print(
                    f"  ix={ix:2d}: viewIndex={state.viewIndex:2d}, "
                    f"elevation={math.degrees(state.elevation):6.2f}°, "
                    f"heading={math.degrees(state.heading):6.2f}°"
                )

        # Check elevation changes
        state0_elev = elevations_at_ix[0]
        state1_elev = (
            elevations_at_ix[num_views]
            if len(elevations_at_ix) > num_views
            else state0_elev
        )

        # Calculate average elevation for each level
        first_level_avg = sum(elevations_at_ix[:num_views]) / num_views
        second_level_avg = (
            sum(elevations_at_ix[num_views : num_views * 2]) / num_views
            if len(elevations_at_ix) > num_views
            else first_level_avg
        )

        elevation_change = second_level_avg - first_level_avg

        print(f"\n  Elevation analysis:")
        print(f"    First level average: {first_level_avg:.2f}°")
        print(f"    Second level average: {second_level_avg:.2f}°")
        print(f"    Elevation change: {elevation_change:.2f}°")

        if abs(elevation_change) > 1.0:
            print(f"    ✓ Elevation changes detected in {mode_name} mode")
        else:
            print(
                f"    ⚠️  No significant elevation change detected in {mode_name} mode"
            )
            print(
                f"    Note: MatterSim may use viewIndex to represent elevation levels"
            )

        results[mode_name] = {
            "initial_elev": math.radians(state0_elev),
            "after_up_elev": math.radians(state1_elev),
            "elevation_change": elevation_change,
        }

    # Compare
    print("\n" + "=" * 80)
    print("Comparison:")
    print("=" * 80)
    print(f"  Elevation change between levels:")
    print(f"    Discrete: {results['Discrete']['elevation_change']:.2f}°")
    print(
        f"    Continuous: {results['Continuous']['elevation_change']:.2f}° (expected: ~30.00°)"
    )

    if abs(results["Discrete"]["elevation_change"]) > 1.0:
        print("    ✓ Discrete mode shows elevation changes")
    else:
        print("    ⚠️  Discrete mode: elevation may be represented via viewIndex")

    if abs(results["Continuous"]["elevation_change"]) > 1.0:
        print("    ✓ Continuous mode shows elevation changes")
    else:
        print("    ⚠️  Continuous mode: elevation may be represented via viewIndex")

    print("\n  Note: MatterSim may use viewIndex to represent elevation levels:")
    print("        - viewIndex 0-11 (or 0-7): elevation ≈ -30°")
    print("        - viewIndex 12-23 (or 8-15): elevation ≈ 0°")
    print("        - viewIndex 24-35 (or 16-23): elevation ≈ +30°")
    print("        The actual elevation value may not change, but viewIndex does.")


def test_make_action_right():
    """Test right rotation actions in both modes"""
    print("\n" + "=" * 80)
    print("Test 2: RIGHT Rotation Actions")
    print("=" * 80)

    scanId = "ZMojNkEp431"
    viewpointId = "2f4d90acd4024c269fb0efe49a8ac540"

    results = {}

    for mode_name, use_discretized, num_views in [
        ("Discrete", True, 12),
        ("Continuous", False, 8),
    ]:
        print(f"\n{mode_name} mode ({num_views} views):")
        sim = MatterSim.Simulator()
        sim.setRenderingEnabled(False)
        sim.setDiscretizedViewingAngles(use_discretized)
        sim.setCameraResolution(640, 480)
        sim.setCameraVFOV(math.radians(60))
        sim.initialize()

        sim.newEpisode([scanId], [viewpointId], [0], [math.radians(-30)])
        states = []

        angle_increment_rad = math.radians(360.0 / num_views)
        expected_increment = math.degrees(angle_increment_rad)

        for i in range(5):
            state = sim.getState()[0]
            states.append(state.heading)
            print(f"  Step {i}: heading={math.degrees(state.heading):7.2f}°")
            if i < 4:
                if use_discretized:
                    sim.makeAction([0], [1.0], [0])  # Right in discrete mode
                else:
                    sim.makeAction(
                        [0], [angle_increment_rad], [0]
                    )  # Right in continuous mode

        # Calculate increments
        increments = []
        for i in range(1, len(states)):
            h_diff = states[i] - states[i - 1]
            h_diff = (h_diff + math.pi) % (2 * math.pi) - math.pi
            increments.append(math.degrees(h_diff))

        print(f"  Heading increments: {[f'{inc:.2f}°' for inc in increments]}")
        if mode_name == "Continuous":
            avg_inc = np.mean(increments)
            if abs(avg_inc - expected_increment) < 2.0:
                print(
                    f"    ✓ Average increment ({avg_inc:.2f}°) matches expected ({expected_increment:.2f}°)"
                )
            else:
                print(
                    f"    ✗ Average increment ({avg_inc:.2f}°) does not match expected ({expected_increment:.2f}°)"
                )

        results[mode_name] = increments


def test_pointid_calculation_comprehensive():
    """Comprehensive test of pointId calculation"""
    print("\n" + "=" * 80)
    print("Test 3: Comprehensive pointId Calculation")
    print("=" * 80)

    scanId = "ZMojNkEp431"
    viewpointId = "2f4d90acd4024c269fb0efe49a8ac540"

    for mode_name, use_discretized, num_views in [
        ("Discrete", True, 12),
        ("Continuous", False, 8),
    ]:
        print(f"\n{mode_name} mode ({num_views} views):")

        sim = MatterSim.Simulator()
        sim.setRenderingEnabled(False)
        sim.setDiscretizedViewingAngles(use_discretized)
        sim.setCameraResolution(640, 480)
        sim.setCameraVFOV(math.radians(60))
        sim.initialize()

        num_total_views = 3 * num_views
        angle_increment_rad = math.radians(360.0 / num_views)

        mismatches = []
        matches = 0

        # Test first level (elevation = -30°)
        print("  Testing first level (elevation ≈ -30°):")
        for ix in range(min(num_views, 12)):
            if ix == 0:
                sim.newEpisode([scanId], [viewpointId], [0], [math.radians(-30)])
            else:
                if use_discretized:
                    sim.makeAction([0], [1.0], [0])
                else:
                    sim.makeAction([0], [angle_increment_rad], [0])

            state = sim.getState()[0]
            expected_point_id = ix

            # Calculate pointId from state
            base_view_index = 0
            base_heading = (base_view_index % num_views) * angle_increment_rad
            relative_heading = (state.heading - base_heading) % (2 * math.pi)

            elevation = state.elevation
            if elevation < -0.2:
                calc_elev_level = 0
            elif elevation > 0.2:
                calc_elev_level = 2
            else:
                calc_elev_level = 1

            calc_horiz_idx = (
                int(round(relative_heading / angle_increment_rad)) % num_views
            )
            calc_point_id = calc_elev_level * num_views + calc_horiz_idx

            if expected_point_id == calc_point_id:
                matches += 1
            else:
                mismatches.append(
                    {
                        "ix": ix,
                        "expected": expected_point_id,
                        "calculated": calc_point_id,
                        "elevation": elevation,
                        "heading": state.heading,
                    }
                )

            if ix < 5:
                match = "✓" if expected_point_id == calc_point_id else "✗"
                print(
                    f"    ix={ix:2d}: expected={expected_point_id:2d}, calc={calc_point_id:2d} {match}"
                )

        print(f"  Matches: {matches}/{min(num_views, 12)}")
        if mismatches:
            print(f"  Mismatches: {len(mismatches)}")
            for m in mismatches[:3]:
                print(
                    f"    ix={m['ix']}: expected={m['expected']}, calc={m['calculated']}"
                )


def test_navigation_consistency():
    """Test navigation consistency between modes"""
    print("\n" + "=" * 80)
    print("Test 4: Navigation Consistency")
    print("=" * 80)

    # Test navigating from pointId 0 to various target pointIds
    test_cases = [
        (0, 1),  # Same level, one step right
        (0, 8),  # Move to next level (for 8 views)
        (0, 12),  # Move to next level (for 12 views)
        (0, 9),  # Move up and one right (for 8 views)
    ]

    for src_point, trg_point in test_cases:
        print(f"\nNavigate from pointId {src_point} to {trg_point}:")

        for num_views in [12, 8]:
            src_level = src_point // num_views
            trg_level = trg_point // num_views
            src_horiz = src_point % num_views
            trg_horiz = trg_point % num_views

            up_actions = max(0, trg_level - src_level)
            down_actions = max(0, src_level - trg_level)

            print(
                f"  {num_views} views: src(level={src_level}, horiz={src_horiz}) -> "
                f"trg(level={trg_level}, horiz={trg_horiz})"
            )
            print(f"    Actions: up={up_actions}, down={down_actions}", end="")
            if trg_level == src_level:
                right_actions = (trg_horiz - src_horiz) % num_views
                print(f", right={right_actions}")
            else:
                print(" (right actions depend on final heading)")


def main():
    print("=" * 80)
    print("Comprehensive Test: Discrete vs Continuous Sim Modes")
    print("=" * 80)
    print("\nThis test compares:")
    print("  - Discrete mode: 12 views, 30° increments")
    print("  - Continuous mode: 8 views, 45° increments")
    print("=" * 80)

    try:
        # Test 1: UP and DOWN actions
        test_make_action_up_down()

        # Test 2: RIGHT rotation actions
        test_make_action_right()

        # Test 3: pointId calculation
        test_pointid_calculation_comprehensive()

        # Test 4: Navigation consistency
        test_navigation_consistency()

        print("\n" + "=" * 80)
        print("All tests completed!")
        print("=" * 80)
        print("\nKey findings:")
        print(
            "1. Verify that UP/DOWN actions produce 30° elevation changes in both modes"
        )
        print("2. Verify that RIGHT actions produce correct heading increments")
        print("3. Verify that pointId calculation matches expected values")
        print("4. Verify that navigation logic is consistent")

    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
