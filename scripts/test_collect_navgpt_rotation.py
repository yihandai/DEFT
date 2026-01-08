#!/usr/bin/env python3
"""
测试脚本：对比我们对NavGPT的feature采集方式（旋转方式和旋转起始点）是否和NavGPT中对observation的description的采集方式相同。

具体做法：
1. 创建./test_collect文件夹
2. 选择几个scan_viewpoint组合
3. 使用我们的方式旋转采集RGB observation
4. 从NavGPT的JSON文件中读取对应的observation description
5. 保存到test_collect/{scan}_{viewpoint}/文件夹下，包含RGB图像和对应的NavGPT description

使用方法：
    # 使用默认参数（8个水平view，24个总view，默认从NavGPT/datasets/R2R/observation_list加载）
    python scripts/test_collect_navgpt_rotation.py
    
    # 使用12个水平view（36个总view）
    python scripts/test_collect_navgpt_rotation.py --panoramic_horizontal_views 12
    
    # 指定NavGPT observation_list目录
    python scripts/test_collect_navgpt_rotation.py \\
        --observation_list_dir NavGPT/datasets/R2R/observation_list
    
    # 指定测试用例
    python scripts/test_collect_navgpt_rotation.py \\
        --test_cases ZMojNkEp431:2f4d90acd4024c269fb0efe49a8ac540 \\
        --panoramic_horizontal_views 8

输出结构：
    test_collect/
        {scan}_{viewpoint}/
            rgb_images/              # RGB图像文件（按旋转顺序）
            states.json              # 每个view的状态信息（heading, elevation, viewIndex）
            navgpt_observation.json  # NavGPT observation描述（从observation_list目录加载）
            comparison.txt           # 对比文件，显示每个图像索引对应的observation
"""

import MatterSim
import os
import sys
import json
import math
import cv2
import numpy as np
import argparse
from pathlib import Path


# # Import config
# from r2r_src.param import args


def get_our_rotation_images(sim, scanId, viewpointId, panoramic_horizontal_views=12):
    """
    使用我们的方式旋转采集RGB图像

    Args:
        sim: MatterSim simulator instance
        scanId: scan ID
        viewpointId: viewpoint ID
        panoramic_horizontal_views: 水平方向的view数量（默认12，对应36个view）

    Returns:
        images: list of RGB images (numpy arrays)
        states: list of state information (heading, elevation, viewIndex)
    """
    images = []
    states = []

    num_total_views = 3 * panoramic_horizontal_views
    angle_increment_rad = math.radians(360.0 / panoramic_horizontal_views)
    use_discretized = panoramic_horizontal_views == 12

    for ix in range(num_total_views):
        if ix == 0:
            # 起始点：heading=0, elevation=-30度
            sim.newEpisode([scanId], [viewpointId], [0], [math.radians(-30)])
        elif ix % panoramic_horizontal_views == 0:
            # 每N个view向上移动30度
            if use_discretized:
                sim.makeAction([0], [1.0], [1.0])
            else:
                sim.makeAction([0], [0.0], [math.radians(30)])
        else:
            # 水平旋转
            if use_discretized:
                sim.makeAction([0], [1.0], [0])
            else:
                sim.makeAction([0], [angle_increment_rad], [0])

        state = sim.getState()[0]

        # 保存图像和状态信息
        # 确保转换为numpy数组（state.rgb可能是内存视图或其他类型）
        rgb_image = np.array(state.rgb, copy=True, dtype=np.uint8)
        images.append(rgb_image)
        states.append(
            {
                "viewIndex": state.viewIndex,
                "heading": state.heading,
                "elevation": state.elevation,
                "ix": ix,
            }
        )

    return images, states


def load_navgpt_observations(observation_list_dir, scan, viewpoint):
    """
    从NavGPT的observation_list目录加载observation description

    Args:
        observation_list_dir: NavGPT observation_list目录 (e.g., NavGPT/datasets/R2R/observation_list)
        scan: scan ID
        viewpoint: viewpoint ID

    Returns:
        obs_data: observation描述数据（从{scanId}.json文件中以{viewpointId}为key获取）
    """
    obs_data = None

    # 加载observation文件：{scanId}.json
    obs_file = os.path.join(observation_list_dir, f"{scan}.json")
    if os.path.exists(obs_file):
        with open(obs_file, "r") as f:
            data = json.load(f)
            if viewpoint in data:
                obs_data = data[viewpoint]
            else:
                print(f"    Warning: Viewpoint {viewpoint} not found in {obs_file}")
    else:
        print(f"    Warning: Observation file not found: {obs_file}")

    return obs_data


def save_collection_results(output_dir, scan, viewpoint, images, states, obs_data):
    """
    保存采集结果到文件

    Args:
        output_dir: 输出目录
        scan: scan ID
        viewpoint: viewpoint ID
        images: RGB图像列表
        states: 状态信息列表
        obs_data: NavGPT observation描述数据（从observation_list目录加载）
    """
    # 创建输出目录
    scan_viewpoint_dir = os.path.join(output_dir, f"{scan}_{viewpoint}")
    os.makedirs(scan_viewpoint_dir, exist_ok=True)

    # 保存RGB图像
    rgb_dir = os.path.join(scan_viewpoint_dir, "rgb_images")
    os.makedirs(rgb_dir, exist_ok=True)

    for ix, (img, state) in enumerate(zip(images, states)):
        # 保存RGB图像
        img_path = os.path.join(
            rgb_dir, f"{ix:02d}_viewIndex{state['viewIndex']:02d}.jpg"
        )
        # 确保img是numpy数组
        if not isinstance(img, np.ndarray):
            img = np.array(img, dtype=np.uint8)
        # MatterSim返回的是RGB格式，需要转换为BGR用于cv2保存
        bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(img_path, bgr_img)

    # 保存状态信息
    states_file = os.path.join(scan_viewpoint_dir, "states.json")
    with open(states_file, "w") as f:
        json.dump(states, f, indent=2)

    # 保存NavGPT observation描述
    if obs_data is not None:
        obs_file = os.path.join(scan_viewpoint_dir, "navgpt_observation.json")
        with open(obs_file, "w") as f:
            json.dump(obs_data, f, indent=2)

    # 创建对比文件（将图像索引和对应的description对应起来）
    comparison_file = os.path.join(scan_viewpoint_dir, "comparison.txt")
    with open(comparison_file, "w") as f:
        f.write(f"Scan: {scan}\n")
        f.write(f"Viewpoint: {viewpoint}\n")
        f.write("=" * 80 + "\n\n")

        for ix, state in enumerate(states):
            f.write(f"Image Index: {ix:02d}\n")
            f.write(f"  ViewIndex: {state['viewIndex']:02d}\n")
            f.write(f"  Heading: {math.degrees(state['heading']):.2f}°\n")
            f.write(f"  Elevation: {math.degrees(state['elevation']):.2f}°\n")

            # 显示对应的NavGPT observation（如果存在）
            if obs_data is not None:
                # obs_data可能是list（按viewIndex顺序）或dict（以viewIndex为key）
                obs_text = None
                if isinstance(obs_data, list) and ix < len(obs_data):
                    obs_text = obs_data[ix]
                elif isinstance(obs_data, dict):
                    # 尝试用viewIndex作为key
                    view_idx = state["viewIndex"]
                    if str(view_idx) in obs_data:
                        obs_text = obs_data[str(view_idx)]
                    elif view_idx in obs_data:
                        obs_text = obs_data[view_idx]
                    elif str(ix) in obs_data:
                        obs_text = obs_data[str(ix)]
                    elif ix in obs_data:
                        obs_text = obs_data[ix]

                if obs_text is not None:
                    obs_str = obs_text if isinstance(obs_text, str) else str(obs_text)
                    # 如果文本太长，截断显示
                    if len(obs_str) > 200:
                        f.write(f"  NavGPT Observation: {obs_str[:200]}...\n")
                    else:
                        f.write(f"  NavGPT Observation: {obs_str}\n")

            f.write("\n")

    print(f"Saved collection results to: {scan_viewpoint_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Test NavGPT rotation collection method"
    )
    parser.add_argument(
        "--panoramic_horizontal_views",
        type=int,
        default=8,
        help="Number of horizontal views (default: 12, use 8 for NavGPT)",
    )
    parser.add_argument(
        "--observation_list_dir",
        type=str,
        default="NavGPT/datasets/R2R/observations_list",
        help="NavGPT observation_list directory (default: NavGPT/datasets/R2R/observation_list). Files are named {scanId}.json with {viewpointId} as keys.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./test_collect",
        help="Output directory for collected data",
    )
    parser.add_argument(
        "--test_cases",
        type=str,
        nargs="+",
        default=None,
        help="Test cases in format 'scan:viewpoint' (e.g., 'ZMojNkEp431:2f4d90acd4024c269fb0efe49a8ac540')",
    )
    parser.add_argument(
        "--vfov",
        type=int,
        default=60,
        help="Vertical field of view in degrees (default: 60)",
    )

    cmd_args = parser.parse_args()

    # 设置panoramic_horizontal_views
    panoramic_horizontal_views = cmd_args.panoramic_horizontal_views

    # 创建输出目录
    output_dir = cmd_args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # 初始化simulator
    sim = MatterSim.Simulator()
    sim.setRenderingEnabled(True)  # 需要渲染才能获取RGB图像
    sim.setDiscretizedViewingAngles(panoramic_horizontal_views == 12)
    sim.setCameraResolution(640, 480)
    sim.setCameraVFOV(math.radians(cmd_args.vfov))
    sim.initialize()

    # 默认测试用例（如果没有提供）
    if cmd_args.test_cases is None:
        # 使用一些常见的测试用例
        # 注意：这些viewpoint ID需要在实际数据中存在
        test_cases = [
            # ("ZMojNkEp431", "2f4d90acd4024c269fb0efe49a8ac540"),
            ("VLzqgDo317F", "fd263d778b534f798d0e1ae48886e5f3"),
            ("VLzqgDo317F", "385019f5d018430fa233d483b253076c"),
            ("8WUmhLawc2A", "62e602fc5b85463dbd8f48ba625d05ef"),
            ("8WUmhLawc2A", "1b48df86b7a149fa8e90161265def866"),
        ]
        print("Using default test cases. Use --test_cases to specify custom cases.")
    else:
        # 解析测试用例
        test_cases = []
        for case in cmd_args.test_cases:
            if ":" in case:
                scan, viewpoint = case.split(":", 1)
                test_cases.append((scan, viewpoint))
            else:
                print(
                    f"Warning: Invalid test case format '{case}', expected 'scan:viewpoint'"
                )

    print(f"Testing {len(test_cases)} scan_viewpoint combinations")
    print(f"Using panoramic_horizontal_views={panoramic_horizontal_views}")
    print(f"Output directory: {output_dir}")

    # 处理每个测试用例
    for scan, viewpoint in test_cases:
        print(f"\nProcessing: {scan} / {viewpoint}")

        try:
            # 1. 使用我们的方式采集RGB图像
            print("  Collecting RGB images using our rotation method...")
            images, states = get_our_rotation_images(
                sim, scan, viewpoint, panoramic_horizontal_views
            )
            print(f"  Collected {len(images)} images")

            # 2. 从NavGPT observation_list目录加载observation描述
            obs_data = None

            if cmd_args.observation_list_dir:
                print("  Loading NavGPT observations...")
                obs_data = load_navgpt_observations(
                    cmd_args.observation_list_dir, scan, viewpoint
                )
                if obs_data is not None:
                    print(f"  Loaded NavGPT observation data")
                else:
                    print(
                        f"  Warning: No observation data found for {scan}/{viewpoint}"
                    )
            else:
                print(
                    "  Warning: NavGPT observation_list directory not provided, skipping observation loading"
                )

            # 3. 保存结果
            print("  Saving results...")
            save_collection_results(
                output_dir, scan, viewpoint, images, states, obs_data
            )

        except Exception as e:
            print(f"  Error processing {scan}/{viewpoint}: {e}")
            import traceback

            traceback.print_exc()

    print(f"\nDone! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
