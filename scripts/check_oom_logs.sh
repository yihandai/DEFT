#!/bin/bash

# Script to check for OOM (Out of Memory) kills and system crashes
# Run this on your Linux server

echo "============================================================"
echo "Checking for OOM (Out of Memory) Kills"
echo "============================================================"
echo ""

echo "1. Checking kernel logs for OOM kills:"
echo "----------------------------------------"
journalctl -k --since "24 hours ago" | grep -i "killed process\|out of memory\|oom\|memory" | tail -30
echo ""

echo "2. Checking for processes killed by OOM:"
echo "----------------------------------------"
journalctl -k --since "24 hours ago" | grep -i "Out of memory: Killed process" | tail -20
echo ""

echo "3. Checking for Python process crashes:"
echo "----------------------------------------"
journalctl --since "24 hours ago" | grep -i "python.*killed\|python.*segfault\|python.*oom" | tail -20
echo ""

echo "4. Checking system reboot times:"
echo "----------------------------------------"
last reboot | head -5
echo ""

echo "5. Current memory status:"
echo "----------------------------------------"
free -h
echo ""

echo "6. Checking for extract_features process:"
echo "----------------------------------------"
ps aux | grep extract_features_24vp | grep -v grep
echo ""

echo "============================================================"
echo "If you see 'Out of memory: Killed process' above,"
echo "that's the cause of your crashes."
echo "============================================================"

