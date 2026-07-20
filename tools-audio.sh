#!/usr/bin/env bash
# Route audio to the first USB soundcard and disable Jetson HDMI and APE.
# Usage: ./tools-audio.sh
#        ./tools-audio.sh --install

# Exit on error, undefined variables, and pipe failure
set -euo pipefail

# Config
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UDEV_RULE_PATH="/etc/udev/rules.d/99-speak-usb-audio.rules"
SETUP_LINK_PATH="/usr/local/bin/speak-usb-audio"
SYSTEM_ASOUND_PATH="/etc/asound.conf"
USER_ASOUND_PATH="${HOME}/.asoundrc"

# Re-exec with sudo when installing system files
if [[ "${1:-}" == "--install" && "${EUID}" -ne 0 ]]; then
    exec sudo --preserve-env=HOME,SUDO_USER bash "${BASH_SOURCE[0]}" "$@"
fi

# Main
main() {
    # Parse command line arguments
    parse_args "$@"

    # Install system hooks
    if [[ "${INSTALL_MODE}" == "true" ]]; then
        install_system
        return 0
    fi

    # Update system config only
    if [[ "${SYSTEM_MODE}" == "true" ]]; then
        configure_system_audio
        return 0
    fi

    configure_audio
}

# Parse command line arguments
parse_args() {
    INSTALL_MODE="false"
    SYSTEM_MODE="false"

    # Read each argument
    for argument in "$@"; do
        if [[ "${argument}" == "--install" ]]; then
            INSTALL_MODE="true"
        elif [[ "${argument}" == "--system" ]]; then
            SYSTEM_MODE="true"
        fi
    done
}

# Return card index for the first USB sound device
find_usb_card() {
    local line card_index

    # Scan proc sound cards for USB audio
    while IFS= read -r line; do
        if [[ "${line}" =~ ^[[:space:]]*([0-9]+)[[:space:]]+.*USB-Audio ]]; then
            card_index="${BASH_REMATCH[1]}"
            if [[ -d "/sys/class/sound/card${card_index}" ]]; then
                echo "${card_index}"
                return 0
            fi
        fi
    done < /proc/asound/cards
    return 1
}

# Write ALSA default config for one card index
write_asoundrc() {
    local card_index="$1"
    local output_path="$2"

    cat > "${output_path}" <<EOF
pcm.!default {
    type plug
    slave.pcm "hw:${card_index},0"
}

ctl.!default {
    type hw
    card ${card_index}
}
EOF
}

# Write system asound config with NVIDIA header
write_system_asound() {
    local card_index="$1"

    cat > "${SYSTEM_ASOUND_PATH}" <<EOF
#
#  ALSA library configuration file
#
#  SPDX-FileCopyrightText: Copyright (c) 2018-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

pcm.!default {
    type plug
    slave {
        pcm "hw:${card_index},0"
        channels 2
    }
    hint.description "USB Audio Device"
}

ctl.!default {
    type hw
    card ${card_index}
}

pcm.demixer {
    type plug
    slave {
        pcm "dmix:${card_index},0"
        channels 2
    }
}
EOF
}

# Turn off non USB pulse cards and pick the USB sink
configure_pulse_runtime() {
    local card sink default_sink_file

    # Skip when pulse is not running
    if ! command -v pactl >/dev/null 2>&1 || ! pactl info >/dev/null 2>&1; then
        return 0
    fi

    # Disable HDMI and APE cards
    while IFS= read -r card; do
        if [[ "${card}" != *usb* ]]; then
            pactl set-card-profile "${card}" off 2>/dev/null || true
        fi
    done < <(pactl list cards short | awk '{print $2}')

    # Pick first USB sink
    sink="$(pactl list sinks short | awk '/usb/ {print $2; exit}')"
    if [[ -z "${sink}" ]]; then
        return 0
    fi

    pactl set-default-sink "${sink}"
    pactl set-sink-volume "${sink}" 100% 2>/dev/null || true
    pactl set-sink-mute "${sink}" 0 2>/dev/null || true

    # Persist default sink for this user session store
    for default_sink_file in "${HOME}/.config/pulse/"*-default-sink; do
        if [[ -f "${default_sink_file}" ]]; then
            echo "${sink}" > "${default_sink_file}"
        fi
    done
}

# Set USB card volume to full
set_usb_volume() {
    local card_index="$1"

    if command -v amixer >/dev/null 2>&1; then
        amixer -c "${card_index}" set PCM 100% unmute >/dev/null 2>&1 || true
        amixer -c "${card_index}" set Master 100% unmute >/dev/null 2>&1 || true
    fi
}

# Update system ALSA config for the current USB card
configure_system_audio() {
    local card_index

    card_index="$(find_usb_card || true)"
    if [[ -z "${card_index}" ]]; then
        exit 0
    fi
    write_system_asound "${card_index}"
}

# Apply ALSA and Pulse settings for the current USB card
configure_audio() {
    local card_index card_name

    card_index="$(find_usb_card || true)"
    if [[ -z "${card_index}" ]]; then
        echo "No USB soundcard found. Plug one in and run ./tools-audio.sh again."
        exit 1
    fi

    card_name="$(cat "/proc/asound/card${card_index}/id")"
    write_asoundrc "${card_index}" "${USER_ASOUND_PATH}"
    set_usb_volume "${card_index}"
    configure_pulse_runtime

    echo "USB card ${card_index}: ${card_name}"
    echo "Wrote ${USER_ASOUND_PATH}"
}

# Install udev rule, then apply settings
install_system() {
    local target_user target_home

    target_user="${SUDO_USER:-${USER}}"
    target_home="$(getent passwd "${target_user}" | cut -d: -f6)"

    ln -sf "${SCRIPT_DIR}/tools-audio.sh" "${SETUP_LINK_PATH}"
    chmod 755 "${SCRIPT_DIR}/tools-audio.sh" "${SETUP_LINK_PATH}"

    cat > "${UDEV_RULE_PATH}" <<EOF
# Reconfigure audio when a USB soundcard is plugged in
ACTION=="add", SUBSYSTEM=="sound", KERNEL=="card*", ENV{ID_BUS}=="usb", RUN+="/bin/sh -c '${SETUP_LINK_PATH} --system; runuser -u ${target_user} -- ${SETUP_LINK_PATH}'"
EOF

    echo "Installed ${UDEV_RULE_PATH}"
    echo "Installed ${SETUP_LINK_PATH}"

    # Apply for the login user, not root
    if [[ -n "${target_home}" && "${target_home}" != "/root" ]]; then
        configure_system_audio
        runuser -u "${target_user}" -- "${SETUP_LINK_PATH}"
    else
        configure_audio
    fi

    udevadm control --reload-rules
    udevadm trigger -s sound
    echo "USB audio setup installed for user ${target_user}."
}

main "$@"
