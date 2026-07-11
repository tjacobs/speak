#!/usr/bin/env bash
# Check or set cpu governor.
# Usage: ./perf.sh

cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# echo ondemand | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_governors
