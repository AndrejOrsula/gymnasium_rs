#!/usr/bin/env bash
set -e

## Configuration
# Default Docker Hub user (used if user is not logged in)
DEFAULT_DOCKERHUB_USER="andrejorsula"

## If the current user is not in the docker group, all docker commands will be run as root
WITH_SUDO=""
if ! grep -qi /etc/group -e "docker.*${USER}"; then
    echo "INFO: The current user ${USER} is not detected in the docker group. All docker commands will be run as root."
    WITH_SUDO="sudo"
fi

## Determine the name of the image to run (automatically inferred from the current user and repository, or using the default if not available)
# Get the current Docker Hub user or use the default
DOCKERHUB_USER="$(${WITH_SUDO} docker info | sed '/Username:/!d;s/.* //')"
DOCKERHUB_USER="${DOCKERHUB_USER:-${DEFAULT_DOCKERHUB_USER}}"
# Get the name of the repository (directory)
SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" &>/dev/null && pwd)"
REPOSITORY_DIR="$(dirname "${SCRIPT_DIR}")"
if [[ -f "${REPOSITORY_DIR}/Dockerfile" ]]; then
    REPOSITORY_NAME="$(basename "${REPOSITORY_DIR}")"
else
    echo >&2 -e "\033[1;31mERROR: Cannot build Docker image because \"${REPOSITORY_DIR}/Dockerfile\" does not exist.\033[0m"
    exit 1
fi
# Combine the user and repository name to form the image name
IMAGE_NAME="${DOCKERHUB_USER}/${REPOSITORY_NAME}"

## Parse TAG and forward additional build arguments
if [ "${#}" -gt "0" ]; then
    if [[ "${1}" != "-"* ]]; then
        IMAGE_NAME="${IMAGE_NAME}:${1}"
        BUILD_ARGS=${*:2}
    else
        BUILD_ARGS=${*:1}
    fi
fi

## Build the image
# shellcheck disable=SC2206
DOCKER_BUILD_CMD=(
    ${WITH_SUDO} docker build
    "${REPOSITORY_DIR}"
    --tag "${IMAGE_NAME}"
    "${BUILD_ARGS}"
)
echo -e "\033[1;90m${DOCKER_BUILD_CMD[*]}\033[0m" | xargs
# shellcheck disable=SC2048
exec ${DOCKER_BUILD_CMD[*]}
