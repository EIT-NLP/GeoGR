#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:-$(pwd)}"
DATA_DIR="${ROOT_DIR}/data"
SCANNET_DIR="${DATA_DIR}/scannet"
SCANS_SRC="${SCANS_SRC:-}"
BASE_URL="${BASE_URL:-https://hf-mirror.com/datasets/zd11024/Video-3D-LLM_data/resolve/main}"

log() {
  echo "[$(date '+%F %T')] $*"
}

get_remote_size() {
  local rel="$1"
  local url="${BASE_URL}/${rel}?download=1"
  local size
  size="$(curl -sIL "${url}" | awk 'tolower($1)=="x-linked-size:"{print $2}' | tr -d '\r' | tail -n1)"
  if [ -z "${size}" ]; then
    size="$(curl -sIL "${url}" | awk 'tolower($1)=="content-length:"{print $2}' | tr -d '\r' | tail -n1)"
  fi
  echo "${size}"
}

file_complete() {
  local rel="$1"
  local local_path="${DATA_DIR}/${rel}"
  if [ ! -f "${local_path}" ]; then
    return 1
  fi
  local remote_size local_size
  remote_size="$(get_remote_size "${rel}")"
  local_size="$(stat -c%s "${local_path}" 2>/dev/null || echo 0)"
  [ -n "${remote_size}" ] && [ "${local_size}" -eq "${remote_size}" ]
}

mkdir -p "${SCANNET_DIR}"

# 1) pcd_with_object_aabbs
if file_complete "pcd_with_object_aabbs.tar.gz"; then
  if [ ! -d "${SCANNET_DIR}/pcd_with_object_aabbs" ]; then
    log "[extract] pcd_with_object_aabbs.tar.gz -> project root (contains data/scannet/...)"
    tar -xzf "${DATA_DIR}/pcd_with_object_aabbs.tar.gz" -C "${ROOT_DIR}"
  else
    log "[skip] data/scannet/pcd_with_object_aabbs already exists"
  fi
else
  log "[wait] pcd_with_object_aabbs.tar.gz not complete yet"
fi

# 2) posed_images split parts -> tar.gz -> extract
parts=(posed_images_part_aa posed_images_part_ab posed_images_part_ac posed_images_part_ad posed_images_part_ae)
all_parts=1
for p in "${parts[@]}"; do
  if ! file_complete "${p}"; then
    all_parts=0
    break
  fi
done

if [ "${all_parts}" -eq 1 ]; then
  if [ ! -d "${SCANNET_DIR}/posed_images" ]; then
    tarball="${DATA_DIR}/posed_images.tar.gz"
    part_sum=0
    for p in "${parts[@]}"; do
      part_sum=$((part_sum + $(stat -c%s "${DATA_DIR}/${p}")))
    done
    tar_size="$(stat -c%s "${tarball}" 2>/dev/null || echo 0)"
    if [ ! -f "${tarball}" ] || [ "${tar_size}" -ne "${part_sum}" ]; then
      log "[merge] posed_images_part_* -> posed_images.tar.gz"
      cat "${DATA_DIR}"/posed_images_part_* > "${tarball}"
    fi
    log "[extract] posed_images.tar.gz -> project root (contains data/scannet/...)"
    tar -xzf "${tarball}" -C "${ROOT_DIR}"
  else
    log "[skip] data/scannet/posed_images already exists"
  fi
else
  log "[wait] posed_images split files are not fully downloaded yet"
fi

# 3) Optional: symlink ScanNet scans from existing local dataset
if [ ! -e "${SCANNET_DIR}/scans" ] && [ -n "${SCANS_SRC}" ] && [ -d "${SCANS_SRC}" ]; then
  ln -s "${SCANS_SRC}" "${SCANNET_DIR}/scans"
  log "[link] data/scannet/scans -> ${SCANS_SRC}"
elif [ -e "${SCANNET_DIR}/scans" ]; then
  log "[skip] data/scannet/scans already exists"
else
  log "[wait] Set SCANS_SRC to an existing ScanNet scans directory."
fi

log "[done] data layout preparation finished"
