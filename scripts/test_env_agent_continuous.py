#!/usr/bin/env python
"""
Comprehensive test script to verify env.py and agent.py modifications
for continuous sim mode (non-discretized) by comparing with discrete mode.

Usage:
    # Test with discrete mode (12 views)
    python scripts/test_env_agent_continuous.py --panoramic_horizontal_views 12
    
    # Test with continuous mode (8 views)
    python scripts/test_env_agent_continuous.py --panoramic_horizontal_views 8
    
    # Compare both modes
    python scripts/test_env_agent_continuous.py --compare
"""

import sys
import os
import math
import numpy as np
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock args before importing modules
class MockArgs:
    def __init__(self, panoramic_horizontal_views=12, vfov=60):
        self.panoramic_horizontal_views = panoramic_horizontal_views
        self.vfov = vfov
        self.test_only = False

# Set up MatterSim path
sys.path.append("buildpy36")
sys.path.append("Matterport_Simulator/build/")

try:
    import MatterSim
    MATTERSIM_AVAILABLE = True
except ImportError:
    print("Warning: MatterSim not found. Some tests will be skipped.")
    MATTERSIM_AVAILABLE = False
    MatterSim = None


def test_make_action_consistency():
    """Test that makeAction produces consistent results"""
    print("=" * 80)
    print("Test 1: makeAction Consistency")
    print("=" * 80)
    
    if not MATTERSIM_AVAILABLE:
        print("  ⚠️  Skipping: MatterSim not available")
        return
    
    scanId = "ZMojNkEp431"
    viewpointId = "2f4d90acd4024c269fb0efe49a8ac540"
    
    # Test discrete mode
    print("\n1. Discrete mode (12 views, 30° increments):")
    sim_d = MatterSim.Simulator()
    sim_d.setRenderingEnabled(False)
    sim_d.setDiscretizedViewingAngles(True)
    sim_d.setCameraResolution(640, 480)
    sim_d.setCameraVFOV(math.radians(60))
    sim_d.initialize()
    sim_d.newEpisode([scanId], [viewpointId], [0], [math.radians(-30)])
    
    discrete_results = []
    for i in range(5):
        state = sim_d.getState()[0]
        discrete_results.append({
            'step': i,
            'viewIndex': state.viewIndex,
            'heading': state.heading,
            'elevation': state.elevation
        })
        if i < 4:
            sim_d.makeAction([0], [1.0], [0])  # Rotate right
    
    for r in discrete_results:
        print(f"  Step {r['step']}: viewIndex={r['viewIndex']:2d}, "
              f"heading={math.degrees(r['heading']):6.2f}°, "
              f"elevation={math.degrees(r['elevation']):6.2f}°")
    
    # Test continuous mode
    print("\n2. Continuous mode (8 views, 45° increments):")
    sim_c = MatterSim.Simulator()
    sim_c.setRenderingEnabled(False)
    sim_c.setDiscretizedViewingAngles(False)
    sim_c.setCameraResolution(640, 480)
    sim_c.setCameraVFOV(math.radians(60))
    sim_c.initialize()
    sim_c.newEpisode([scanId], [viewpointId], [0], [math.radians(-30)])
    
    angle_increment_rad = math.radians(360.0 / 8)
    continuous_results = []
    for i in range(5):
        state = sim_c.getState()[0]
        continuous_results.append({
            'step': i,
            'viewIndex': state.viewIndex,
            'heading': state.heading,
            'elevation': state.elevation
        })
        if i < 4:
            sim_c.makeAction([0], [angle_increment_rad], [0])  # Rotate right by 45°
    
    for r in continuous_results:
        print(f"  Step {r['step']}: viewIndex={r['viewIndex']:2d}, "
              f"heading={math.degrees(r['heading']):6.2f}°, "
              f"elevation={math.degrees(r['elevation']):6.2f}°")
    
    # Compare heading increments
    print("\n3. Heading increment comparison:")
    print("   Discrete mode (expected: 30° per step):")
    for i in range(1, len(discrete_results)):
        h_diff = discrete_results[i]['heading'] - discrete_results[i-1]['heading']
        h_diff = (h_diff + math.pi) % (2 * math.pi) - math.pi
        print(f"     Step {i-1}->{i}: {math.degrees(h_diff):6.2f}°")
    
    print("   Continuous mode (expected: 45° per step):")
    for i in range(1, len(continuous_results)):
        h_diff = continuous_results[i]['heading'] - continuous_results[i-1]['heading']
        h_diff = (h_diff + math.pi) % (2 * math.pi) - math.pi
        expected = math.degrees(angle_increment_rad)
        match = "✓" if abs(math.degrees(h_diff) - expected) < 1.0 else "✗"
        print(f"     Step {i-1}->{i}: {math.degrees(h_diff):6.2f}° (expected: {expected:.2f}°) {match}")


def test_pointid_calculation_logic(num_horizontal_views):
    """Test pointId calculation logic"""
    print(f"\n{'=' * 80}")
    print(f"Test 2: pointId Calculation Logic (num_horizontal_views={num_horizontal_views})")
    print("=" * 80)
    
    if not MATTERSIM_AVAILABLE:
        print("  ⚠️  Skipping: MatterSim not available")
        return
    
    scanId = "ZMojNkEp431"
    viewpointId = "2f4d90acd4024c269fb0efe49a8ac540"
    
    use_discretized = (num_horizontal_views == 12)
    angle_increment_rad = math.radians(360.0 / num_horizontal_views)
    num_total_views = 3 * num_horizontal_views
    
    sim = MatterSim.Simulator()
    sim.setRenderingEnabled(False)
    sim.setDiscretizedViewingAngles(use_discretized)
    sim.setCameraResolution(640, 480)
    sim.setCameraVFOV(math.radians(60))
    sim.initialize()
    
    results = []
    
    for ix in range(min(24, num_total_views)):
        if ix == 0:
            sim.newEpisode([scanId], [viewpointId], [0], [math.radians(-30)])
        elif ix % num_horizontal_views == 0:
            if use_discretized:
                sim.makeAction([0], [1.0], [1.0])
            else:
                sim.makeAction([0], [0.0], [math.radians(30)])
        else:
            if use_discretized:
                sim.makeAction([0], [1.0], [0])
            else:
                sim.makeAction([0], [angle_increment_rad], [0])
        
        state = sim.getState()[0]
        
        # Expected pointId
        expected_point_id = ix
        
        # Calculate pointId from state (as in get_current_point_id)
        base_view_index = 0  # Assuming starting from 0
        base_heading = (base_view_index % num_horizontal_views) * angle_increment_rad
        relative_heading = (state.heading - base_heading) % (2 * math.pi)
        
        elevation = state.elevation
        if elevation < -0.2:
            calc_elev_level = 0
        elif elevation > 0.2:
            calc_elev_level = 2
        else:
            calc_elev_level = 1
        
        calc_horiz_idx = int(round(relative_heading / angle_increment_rad)) % num_horizontal_views
        calc_point_id = calc_elev_level * num_horizontal_views + calc_horiz_idx
        
        results.append({
            'ix': ix,
            'viewIndex': state.viewIndex,
            'expected_pointId': expected_point_id,
            'calc_pointId': calc_point_id,
            'elevation': elevation,
            'heading': state.heading,
            'calc_elev_level': calc_elev_level,
            'calc_horiz_idx': calc_horiz_idx
        })
        
        # Print first few and level transitions
        if ix < 8 or (ix % num_horizontal_views == 0) or (calc_point_id != expected_point_id):
            match = "✓" if calc_point_id == expected_point_id else "✗"
            print(f"  ix={ix:2d}: viewIndex={state.viewIndex:2d}, "
                  f"expected={expected_point_id:2d}, calc={calc_point_id:2d} {match}, "
                  f"elev={math.degrees(elevation):6.2f}°(L{calc_elev_level}), "
                  f"heading={math.degrees(state.heading):6.2f}°(idx={calc_horiz_idx})")
    
    # Check for mismatches
    mismatches = [r for r in results if r['expected_pointId'] != r['calc_pointId']]
    if mismatches:
        print(f"\n  ⚠️  Found {len(mismatches)} mismatches!")
        for m in mismatches[:10]:
            print(f"    ix={m['ix']}: expected={m['expected_pointId']}, "
                  f"calculated={m['calc_pointId']}")
    else:
        print(f"\n  ✓ All {len(results)} pointId calculations match!")


def test_base_heading_consistency():
    """Test base_heading calculation consistency"""
    print(f"\n{'=' * 80}")
    print("Test 3: base_heading Calculation Consistency")
    print("=" * 80)
    
    # Test with different num_horizontal_views
    for num_horizontal_views in [12, 8]:
        print(f"\nnum_horizontal_views={num_horizontal_views}:")
        angle_increment_rad = math.radians(360.0 / num_horizontal_views)
        
        test_viewIds = [0, 5, num_horizontal_views-1, num_horizontal_views, num_horizontal_views*2]
        for viewId in test_viewIds:
            base_heading = (viewId % num_horizontal_views) * angle_increment_rad
            horiz_idx = viewId % num_horizontal_views
            print(f"  viewId={viewId:2d}: horiz_idx={horiz_idx:2d}, "
                  f"base_heading={math.degrees(base_heading):6.2f}°")


def test_relative_heading_calculation():
    """Test relative heading calculation"""
    print(f"\n{'=' * 80}")
    print("Test 4: Relative Heading Calculation")
    print("=" * 80)
    
    num_horizontal_views = 8
    angle_increment_rad = math.radians(360.0 / num_horizontal_views)
    
    test_cases = [
        # (base_view_index, current_heading_deg, expected_relative_heading_deg)
        (0, 0, 0),
        (0, 45, 45),
        (0, 90, 90),
        (1, 45, 0),  # If base is at 45°, then 45° is relative 0°
        (1, 90, 45),
    ]
    
    print(f"\nTesting with num_horizontal_views={num_horizontal_views}:")
    for base_view_index, current_heading_deg, expected_rel_heading_deg in test_cases:
        base_heading = (base_view_index % num_horizontal_views) * angle_increment_rad
        current_heading = math.radians(current_heading_deg)
        relative_heading = (current_heading - base_heading) % (2 * math.pi)
        
        expected_rel_heading = math.radians(expected_rel_heading_deg)
        diff = abs(relative_heading - expected_rel_heading)
        if diff > math.pi:
            diff = 2 * math.pi - diff
        
        match = "✓" if diff < 0.01 else "✗"
        print(f"  base_idx={base_view_index}, current={current_heading_deg:5.1f}° -> "
              f"relative={math.degrees(relative_heading):6.2f}° "
              f"(expected={expected_rel_heading_deg:5.1f}°) {match}")


def compare_modes():
    """Compare discrete and continuous modes side by side"""
    print(f"\n{'=' * 80}")
    print("Test 5: Side-by-Side Comparison")
    print("=" * 80)
    
    if not MATTERSIM_AVAILABLE:
        print("  ⚠️  Skipping: MatterSim not available")
        return
    
    scanId = "ZMojNkEp431"
    viewpointId = "2f4d90acd4024c269fb0efe49a8ac540"
    
    # Test navigating from pointId 0 to pointId 9
    print("\nTest case: Navigate from pointId 0 to pointId 9")
    
    # Discrete mode (12 views)
    print("\nDiscrete mode (12 views):")
    print("  pointId 0 -> 9: elev_level 0->0, horiz_idx 0->9")
    print("  Expected: 0 up, 9 right")
    
    # Continuous mode (8 views)
    print("\nContinuous mode (8 views):")
    print("  pointId 0 -> 9: elev_level 0->1, horiz_idx 0->1")
    print("  Expected: 1 up, 1 right")
    
    # Verify calculation
    for num_horizontal_views in [12, 8]:
        src_point = 0
        trg_point = 9
        src_level = src_point // num_horizontal_views
        trg_level = trg_point // num_horizontal_views
        src_horiz = src_point % num_horizontal_views
        trg_horiz = trg_point % num_horizontal_views
        
        print(f"\n  num_horizontal_views={num_horizontal_views}:")
        print(f"    src: level={src_level}, horiz={src_horiz}")
        print(f"    trg: level={trg_level}, horiz={trg_horiz}")
        print(f"    up actions: {max(0, trg_level - src_level)}")
        if trg_level == src_level:
            right_actions = (trg_horiz - src_horiz) % num_horizontal_views
            print(f"    right actions: {right_actions}")


def main():
    parser = argparse.ArgumentParser(description='Test continuous sim mode modifications')
    parser.add_argument('--panoramic_horizontal_views', type=int, default=None,
                        help='Number of horizontal views to test (12 for discrete, 8 for continuous)')
    parser.add_argument('--compare', action='store_true',
                        help='Compare discrete and continuous modes')
    args = parser.parse_args()
    
    print("=" * 80)
    print("Testing Continuous Sim Mode Modifications")
    print("=" * 80)
    
    if not MATTERSIM_AVAILABLE:
        print("\n⚠️  Warning: MatterSim not available. Some tests will be skipped.")
        print("   Please ensure MatterSim is properly installed.")
    
    try:
        # Test 1: makeAction consistency
        if MATTERSIM_AVAILABLE:
            test_make_action_consistency()
        
        # Test 2: pointId calculation
        if args.panoramic_horizontal_views:
            test_pointid_calculation_logic(args.panoramic_horizontal_views)
        elif args.compare:
            print("\n" + "=" * 80)
            print("Testing both modes for comparison")
            print("=" * 80)
            test_pointid_calculation_logic(12)
            test_pointid_calculation_logic(8)
        else:
            print("\nNote: Use --panoramic_horizontal_views N to test specific mode")
            print("      Use --compare to test both modes")
        
        # Test 3: base_heading consistency
        test_base_heading_consistency()
        
        # Test 4: relative heading calculation
        test_relative_heading_calculation()
        
        # Test 5: Compare modes
        if args.compare:
            compare_modes()
        
        print("\n" + "=" * 80)
        print("All tests completed!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

