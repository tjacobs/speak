#!/usr/bin/env bash
# Check or set cpu governor.
# Usage: ./tools-perf.sh

echo "Current:"
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

echo "Available:"
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_governors

echo "Set: "
echo "echo ondemand | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor"
