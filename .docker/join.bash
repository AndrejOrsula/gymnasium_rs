#!/usr/bin/env bash
set -e

## Configuration
# Default repository name (used if inferred name cannot be determined)
DEFAULT_REPOSITORY_NAME="gymnasium_rs"
# Flags for executing a command inside the container
DOCKER_EXEC_OPTS="${DOCKER_EXEC_OPTS:-
    --interactive
    --tty
}"
# Default command to execute inside the container
DEFAULT_CMD="bash"

## If the current user is not in the docker group, all docker commands will be run as root
WITH_SUDO=""
if ! grep -qi /etc/group -e "docker.*${USER}"; then
    echo "INFO: The current user ${USER} is not detected in the docker group. All docker commands will be run as root."
    WITH_SUDO="sudo"
fi

## Get the name of the repository (directory) or use the default
SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" &>/dev/null && pwd)"
REPOSITORY_DIR="$(dirname "${SCRIPT_DIR}")"
if [[ -f "${REPOSITORY_DIR}/Dockerfile" ]]; then
    CONTAINER_NAME="$(basename "${REPOSITORY_DIR}")"
else
    CONTAINER_NAME="${DEFAULT_REPOSITORY_NAME}"
fi

## Parse CMD if provided
if [ "${#}" -gt "0" ]; then
    # If the first argument is a positive integer, it is parsed as the suffix of the container name
    if [[ "${1}" =~ ^[0-9]+$ ]]; then
        CONTAINER_NAME_SUFFIX="${1}"
        if [ "${#}" -gt "1" ]; then
            CMD=${*:2}
        else
            CMD="${DEFAULT_CMD}"
        fi
    else
        CMD=${*:1}
    fi
else
    CMD="${DEFAULT_CMD}"
fi

## Verify that the container is active
RUNNING_CONTAINERS=$(${WITH_SUDO} docker container list --format "{{.Names}}" | grep -i "${CONTAINER_NAME}" || :)
RUNNING_CONTAINERS_COUNT=$(echo "${RUNNING_CONTAINERS}" | wc -w)
if [ "${RUNNING_CONTAINERS_COUNT}" -eq "0" ]; then
    echo >&2 -e "\033[1;31mERROR: There are no active containers with the name \"${CONTAINER_NAME}\". Start the container first before attempting to join it.\033[0m"
    exit 1
fi

print_running_containers_and_usage() {
    RUNNING_CONTAINERS=$(echo "${RUNNING_CONTAINERS}" | sort --version-sort)
    echo >&2 "Active *${CONTAINER_NAME}* containers:"
    i=0
    echo "${RUNNING_CONTAINERS}" | while read -r line; do
        echo >&2 -e "\t${i}: ${line}"
        i=$((i + 1))
    done
    echo >&2 "Usage: ${0} [CONTAINER_NAME_SUFFIX] [CMD]"
}
## If provided, append the numerical suffix to the container name
if [[ -n "${CONTAINER_NAME_SUFFIX}" ]]; then
    if [ "${CONTAINER_NAME_SUFFIX}" -eq "0" ]; then
        CONTAINER_NAME_SUFFIX=""
    fi
    # Make sure that the container with the specified suffix is active
    if ! echo "${RUNNING_CONTAINERS}" | grep -qi "${CONTAINER_NAME}${CONTAINER_NAME_SUFFIX}"; then
        echo >&2 -e "\033[1;31mERROR: Invalid argument \"${CONTAINER_NAME_SUFFIX}\" — there is no active container with the name \"${CONTAINER_NAME}${CONTAINER_NAME_SUFFIX}\".\033[0m"
        print_running_containers_and_usage
        exit 2
    fi
    CONTAINER_NAME="${CONTAINER_NAME}${CONTAINER_NAME_SUFFIX}"
else
    # Otherwise, check if there is only one active container with the specified name
    if [ "${RUNNING_CONTAINERS_COUNT}" -gt "1" ]; then
        echo >&2 -e "\033[1;31mERROR: More than one active *${CONTAINER_NAME}* container. Specify the suffix of the container name as the first argument.\033[0m"
        print_running_containers_and_usage
        exit 2
    else
        # If there is only one active container, use it regardless of the suffix
        CONTAINER_NAME="${RUNNING_CONTAINERS}"
    fi
fi

## Execute command inside the container
# shellcheck disable=SC2206
DOCKER_EXEC_CMD=(
    ${WITH_SUDO} docker exec
    "${DOCKER_EXEC_OPTS}"
    "${CONTAINER_NAME}"
    "${CMD}"
)
echo -e "\033[1;90m${DOCKER_EXEC_CMD[*]}\033[0m" | xargs
# shellcheck disable=SC2048
exec ${DOCKER_EXEC_CMD[*]}
