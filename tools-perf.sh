#!/usr/bin/env bash
# Check or set cpu scaling mode.
# Usage: ./tools-perf.sh

AVAILABLE='/sys/devices/system/cpu/cpu0/cpufreq/scaling_available_governors'

echo "Current:"
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

echo "Available:"
cat "${AVAILABLE}"

echo "Set: "
echo "echo ondemand | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor"
