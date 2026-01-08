#!/usr/bin/env python3
"""
Script to parse reward information from log files and plot average reward curves.

Parses lines like:
    total reward [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]

Calculates the average reward for each episode and plots a curve.
"""

import re
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple

# ============================================================================
# Configuration - Edit these variables as needed
# ============================================================================
# LOG_FILE = "logs/navgpt/train_mask_025_2_2025_12_21.log"
# LOG_FILE = "logs/navgpt/train_mask_05_2025_12_20.log"
LOG_FILE = "logs/navgpt2/mask/train_mask_2025_12_30.log"
OUTPUT_FILE = "./logs/navgpt2/mask/reward_curve.png"  # Output file path (e.g., "rewards.png") or None to display interactively
PLOT_TITLE = "Reward Curve"  # Title for the plot (None uses default)
SHOW_PLOT = True  # Set to False to only print summary statistics without plotting
SMOOTH_CURVE = True  # Set to True to smooth the curve using moving average
SMOOTH_WINDOW = 10  # Window size for moving average smoothing (larger = smoother)
# ============================================================================
# Show help message and exit
if len(sys.argv) > 1 and sys.argv[1] == "--help":
    print("Usage: python plot_rewards.py [--help]")
    print("Options:")
    print("  --help: Show this help message and exit")
    sys.exit(0)


def parse_rewards_from_log(log_file_path: str) -> List[Tuple[int, float]]:
    """
    Parse reward information from log file.

    Args:
        log_file_path: Path to the log file

    Returns:
        List of tuples (episode_number, average_reward)
    """
    rewards_data = []

    # Pattern to match: "total reward [0.0, 0.0, 1.0, ...]"
    reward_pattern = re.compile(r"total reward\s+\[([\d\.,\s]+)\]")

    # Pattern to match episode numbers (lines with just a number)
    episode_pattern = re.compile(r"^\s*(\d+)\s*$")

    try:
        with open(log_file_path, "r") as f:
            current_episode = None
            for line in f:
                # Check for episode number
                episode_match = episode_pattern.match(line.strip())
                if episode_match:
                    current_episode = int(episode_match.group(1))
                    continue

                # Check for reward list
                reward_match = reward_pattern.search(line)
                if reward_match:
                    # Extract the list string and parse it
                    reward_list_str = reward_match.group(1)
                    # Parse the list of floats
                    try:
                        reward_list = [
                            float(x.strip()) for x in reward_list_str.split(",")
                        ]
                        if reward_list:
                            avg_reward = sum(reward_list) / len(reward_list)
                            # Use current episode if available, otherwise use length of rewards_data + 1
                            episode = (
                                current_episode
                                if current_episode is not None
                                else len(rewards_data) + 1
                            )
                            rewards_data.append((episode, avg_reward))
                            current_episode = None  # Reset after processing
                    except ValueError as e:
                        print(
                            f"Warning: Could not parse reward list '{reward_list_str}': {e}",
                            file=sys.stderr,
                        )
                        continue

    except FileNotFoundError:
        print(f"Error: File '{log_file_path}' not found.", file=sys.stderr)
        return []
    except Exception as e:
        print(f"Error reading file: {e}", file=sys.stderr)
        return []

    return rewards_data


def smooth_curve(data: List[float], window_size: int) -> List[float]:
    """
    Apply moving average smoothing to the data.

    Args:
        data: List of values to smooth
        window_size: Size of the moving average window

    Returns:
        Smoothed data list
    """
    if len(data) < window_size:
        return data

    smoothed = []
    half_window = window_size // 2

    for i in range(len(data)):
        # Calculate the window boundaries
        start = max(0, i - half_window)
        end = min(len(data), i + half_window + 1)

        # Calculate the average within the window
        window_data = data[start:end]
        smoothed.append(sum(window_data) / len(window_data))

    return smoothed


def plot_reward_curve(
    rewards_data: List[Tuple[int, float]],
    output_file: str = None,
    title: str = None,
    smooth: bool = False,
    smooth_window: int = 10,
):
    """
    Plot the average reward curve.

    Args:
        rewards_data: List of tuples (episode_number, average_reward)
        output_file: Optional path to save the plot (if None, displays interactively)
        title: Optional title for the plot
        smooth: Whether to smooth the curve
        smooth_window: Window size for smoothing
    """
    if not rewards_data:
        print("No reward data to plot.", file=sys.stderr)
        return

    episodes = [ep for ep, _ in rewards_data]
    avg_rewards = [reward for _, reward in rewards_data]

    plt.figure(figsize=(10, 6))

    # Apply smoothing if requested
    if smooth and len(avg_rewards) > smooth_window:
        smoothed_rewards = smooth_curve(avg_rewards, smooth_window)
        # Plot both original and smoothed curves
        plt.plot(
            episodes,
            avg_rewards,
            marker="o",
            linestyle="-",
            linewidth=1.0,
            markersize=3,
            alpha=0.3,
            label="Original",
            color="lightblue",
        )
        plt.plot(
            episodes,
            smoothed_rewards,
            marker="o",
            linestyle="-",
            linewidth=2.0,
            markersize=4,
            label="Smoothed",
            color="blue",
        )
        plt.legend()
    else:
        plt.plot(
            episodes,
            avg_rewards,
            marker="o",
            linestyle="-",
            linewidth=1.5,
            markersize=4,
        )
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Average Reward", fontsize=12)
    plt.title(title or "Average Reward per Episode", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Add statistics text box
    mean_reward = np.mean(avg_rewards)
    std_reward = np.std(avg_rewards)
    max_reward = np.max(avg_rewards)
    min_reward = np.min(avg_rewards)

    stats_text = f"Mean: {mean_reward:.4f}\nStd: {std_reward:.4f}\nMax: {max_reward:.4f}\nMin: {min_reward:.4f}"
    plt.text(
        0.02,
        0.98,
        stats_text,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {output_file}")
    else:
        plt.show()

    plt.close()


def print_summary(rewards_data: List[Tuple[int, float]]):
    """Print a summary of the reward data."""
    if not rewards_data:
        return

    episodes = [ep for ep, _ in rewards_data]
    avg_rewards = [reward for _, reward in rewards_data]

    print("=" * 70)
    print("Reward Analysis Summary")
    print("=" * 70)
    print(f"Total episodes: {len(rewards_data)}")
    print(f"Mean reward: {np.mean(avg_rewards):.6f}")
    print(f"Std reward: {np.std(avg_rewards):.6f}")
    print(
        f"Max reward: {np.max(avg_rewards):.6f} (Episode {episodes[np.argmax(avg_rewards)]})"
    )
    print(
        f"Min reward: {np.min(avg_rewards):.6f} (Episode {episodes[np.argmin(avg_rewards)]})"
    )
    print("=" * 70)


if __name__ == "__main__":
    # Parse rewards from log file
    print(f"Parsing rewards from: {LOG_FILE}")
    rewards_data = parse_rewards_from_log(LOG_FILE)

    if not rewards_data:
        print("No reward data found in the log file.", file=sys.stderr)
        sys.exit(1)

    # Print summary
    print_summary(rewards_data)

    # Plot if requested
    if SHOW_PLOT:
        plot_reward_curve(
            rewards_data,
            output_file=OUTPUT_FILE,
            title=PLOT_TITLE,
            smooth=SMOOTH_CURVE,
            smooth_window=SMOOTH_WINDOW,
        )
