#!/usr/bin/env python3

"""Script to check system crash logs and identify the cause of system shutdown/crash.
Works on both macOS and Linux systems."""

import os
import sys
import subprocess
import platform
from datetime import datetime, timedelta


def run_command(cmd, shell=False):
    """Run a shell command and return output"""
    try:
        result = subprocess.run(
            cmd, shell=shell, capture_output=True, text=True, timeout=30
        )
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return "", "Command timed out", 1
    except Exception as e:
        return "", str(e), 1


def check_macos_logs():
    """Check macOS system logs for crash/shutdown reasons"""
    print("=" * 70)
    print("Checking macOS System Logs")
    print("=" * 70)
    print()

    # Check system.log for recent shutdowns
    print("1. Recent shutdown events (last 24 hours):")
    print("-" * 70)
    cmd = [
        "log",
        "show",
        "--predicate",
        'eventMessage contains "shutdown" OR eventMessage contains "reboot" OR eventMessage contains "halt"',
        "--last",
        "24h",
        "--style",
        "compact",
    ]
    stdout, stderr, code = run_command(cmd)
    if stdout:
        lines = stdout.strip().split("\n")
        for line in lines[-20:]:  # Show last 20 lines
            print(line)
    else:
        print("  No recent shutdown events found")
    print()

    # Check for kernel panics
    print("2. Recent kernel panics:")
    print("-" * 70)
    panic_dir = "/Library/Logs/DiagnosticReports"
    if os.path.exists(panic_dir):
        panic_files = [
            f
            for f in os.listdir(panic_dir)
            if f.startswith("kernel_") and f.endswith(".panic")
        ]
        panic_files.sort(reverse=True)
        if panic_files:
            print(f"  Found {len(panic_files)} panic reports")
            print("  Most recent panics:")
            for f in panic_files[:5]:
                filepath = os.path.join(panic_dir, f)
                mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
                print(f"    {f} - {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
                # Show first few lines
                try:
                    with open(filepath, "r") as pf:
                        lines = pf.readlines()[:10]
                        for line in lines:
                            if line.strip():
                                print(f"      {line.strip()}")
                except:
                    pass
        else:
            print("  No kernel panic reports found")
    else:
        print(f"  Panic directory not found: {panic_dir}")
    print()

    # Check for OOM (Out of Memory) kills
    print("3. Out of Memory (OOM) events:")
    print("-" * 70)
    cmd = [
        "log",
        "show",
        "--predicate",
        'eventMessage contains "low swap" OR eventMessage contains "memory pressure" OR eventMessage contains "killed process"',
        "--last",
        "24h",
        "--style",
        "compact",
    ]
    stdout, stderr, code = run_command(cmd)
    if stdout:
        lines = stdout.strip().split("\n")
        for line in lines[-20:]:
            print(line)
    else:
        print("  No OOM events found in system logs")
    print()

    # Check console logs
    print("4. Recent console errors:")
    print("-" * 70)
    cmd = [
        "log",
        "show",
        "--predicate",
        'messageType == "Error" OR messageType == "Fault"',
        "--last",
        "6h",
        "--style",
        "compact",
    ]
    stdout, stderr, code = run_command(cmd)
    if stdout:
        lines = stdout.strip().split("\n")
        for line in lines[-30:]:
            print(line)
    else:
        print("  No recent errors found")
    print()


def check_linux_logs():
    """Check Linux system logs for crash/shutdown reasons"""
    print("=" * 70)
    print("Checking Linux System Logs")
    print("=" * 70)
    print()

    # Check dmesg for OOM kills and kernel panics
    print("1. Kernel messages (dmesg) - OOM kills and panics:")
    print("-" * 70)
    # Try dmesg with sudo first, then without
    cmd = "dmesg -T 2>/dev/null | tail -100"
    stdout, stderr, code = run_command(cmd, shell=True)
    if code != 0 or not stdout:
        # Try without -T flag (older systems)
        cmd = "dmesg 2>/dev/null | tail -100"
        stdout, stderr, code = run_command(cmd, shell=True)
    if code != 0 or not stdout:
        # Try with sudo
        cmd = "sudo dmesg -T 2>/dev/null | tail -100"
        stdout, stderr, code = run_command(cmd, shell=True)

    if stdout:
        lines = stdout.strip().split("\n")
        relevant_lines = [
            line
            for line in lines
            if "killed process" in line.lower()
            or "oom" in line.lower()
            or "panic" in line.lower()
            or "segfault" in line.lower()
            or "out of memory" in line.lower()
        ]
        if relevant_lines:
            for line in relevant_lines[-20:]:
                print(line)
        else:
            print("  No OOM or panic messages found in recent dmesg")
            print("  (Note: dmesg may require sudo privileges)")
    else:
        print("  Could not access dmesg (may require sudo privileges)")
        print("  Try running manually: sudo dmesg | grep -i 'killed\|oom'")
    print()

    # Check journalctl for system crashes
    print("2. System journal (journalctl) - Recent crashes:")
    print("-" * 70)
    cmd = "journalctl -k --since '24 hours ago' | grep -i 'killed\|oom\|panic\|segfault' | tail -30"
    stdout, stderr, code = run_command(cmd, shell=True)
    if stdout:
        print(stdout)
    else:
        print("  No crash-related messages found in journal")
    print()

    # Check for OOM kills specifically
    print("3. OOM Killer events:")
    print("-" * 70)
    cmd = "journalctl -k --since '24 hours ago' | grep -i 'out of memory\|killed process' | tail -20"
    stdout, stderr, code = run_command(cmd, shell=True)
    if stdout:
        print(stdout)
    else:
        print("  No OOM kill events found")
    print()

    # Check system shutdown reasons
    print("4. System shutdown/reboot events:")
    print("-" * 70)
    cmd = "journalctl --since '24 hours ago' | grep -i 'shutdown\|reboot\|halt\|power' | tail -20"
    stdout, stderr, code = run_command(cmd, shell=True)
    if stdout:
        print(stdout)
    else:
        print("  No shutdown events found")
    print()

    # Check last boot time
    print("5. Last boot time:")
    print("-" * 70)
    cmd = "who -b"
    stdout, stderr, code = run_command(cmd, shell=True)
    if stdout:
        print(stdout)
    print()

    # Check system memory info
    print("6. System memory information:")
    print("-" * 70)
    cmd = "free -h"
    stdout, stderr, code = run_command(cmd, shell=True)
    if stdout:
        print(stdout)
    print()


def check_process_logs():
    """Check for Python process crashes"""
    print("=" * 70)
    print("Checking for Python Process Crashes")
    print("=" * 70)
    print()

    system = platform.system()
    if system == "Darwin":  # macOS
        print("Searching for Python crash reports...")
        crash_dir = os.path.expanduser("~/Library/Logs/DiagnosticReports")
        if os.path.exists(crash_dir):
            crash_files = [
                f
                for f in os.listdir(crash_dir)
                if "Python" in f or "python" in f or "extract_features" in f
            ]
            crash_files.sort(reverse=True)
            if crash_files:
                print(f"  Found {len(crash_files)} Python-related crash reports:")
                for f in crash_files[:10]:
                    filepath = os.path.join(crash_dir, f)
                    mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
                    print(f"    {f} - {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                print("  No Python crash reports found")
        else:
            print(f"  Crash directory not found: {crash_dir}")

    elif system == "Linux":
        print("Checking for Python segfaults in system logs...")
        cmd = "journalctl --since '24 hours ago' | grep -i 'python.*segfault\|python.*killed' | tail -20"
        stdout, stderr, code = run_command(cmd, shell=True)
        if stdout:
            print(stdout)
        else:
            print("  No Python crash events found in logs")

    print()


def main():
    """Main function"""
    system = platform.system()
    print(f"Detected system: {system}")
    print()

    if system == "Darwin":
        check_macos_logs()
    elif system == "Linux":
        check_linux_logs()
    else:
        print(f"Unsupported system: {system}")
        print("Please check logs manually:")
        print("  - macOS: Console.app or 'log show' command")
        print("  - Linux: journalctl, dmesg, /var/log/syslog")

    check_process_logs()

    print("=" * 70)
    print("Manual Check Commands:")
    print("=" * 70)
    if system == "Darwin":
        print("macOS:")
        print("  1. Open Console.app and search for 'shutdown', 'panic', 'killed'")
        print("  2. Check /Library/Logs/DiagnosticReports/ for kernel panics")
        print(
            "  3. Run: log show --predicate 'eventMessage contains \"killed\"' --last 24h"
        )
        print("  4. Check Activity Monitor for memory pressure")
    elif system == "Linux":
        print("Linux:")
        print("  1. Check OOM kills:")
        print("     sudo dmesg | grep -i 'killed process'")
        print("     sudo dmesg -T | tail -100 | grep -i 'killed\|oom'")
        print("  2. Check system logs (no sudo needed):")
        print("     journalctl -k --since '24 hours ago' | grep -i 'killed\|oom'")
        print("  3. Check memory: free -h")
        print("  4. Check last boot: who -b")
        print("  5. Check for Python crashes:")
        print("     journalctl --since '24 hours ago' | grep -i python")
        print("  6. Check system logs file (if journalctl not available):")
        print("     sudo tail -100 /var/log/syslog | grep -i 'killed\|oom'")
        print("     sudo tail -100 /var/log/messages | grep -i 'killed\|oom'")
    print()


if __name__ == "__main__":
    main()
