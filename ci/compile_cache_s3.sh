#!/usr/bin/env bash
# Shared S3 layer under the GPU CI kernel compile cache (see ci/bash.sh).
#
#   compile_cache_s3.sh restore <https-base> <image> <lane> <dir> [--writer]
#   compile_cache_s3.sh upload  <s3-base>    <image> <lane> <dir>
#
# The cache is one gzipped tarball per (CI image, GPU, lane), stored at
# <base>/<image key>/<lane>.tar.gz under a prefix that only pushes to main can
# write. Any job restores it over anonymous HTTPS:
#
# - Merge mode (pull requests): add what the per-runner cache in <dir> lacks.
#   A CuTe-DSL module directory is one unit, because its single meta.json
#   vouches for every object in it: a module the runner does not have is
#   moved in whole; one it has gets missing objects only when both meta.json
#   files are identical. Every other artifact is keyed by its path, so missing
#   files are copied and existing ones kept. The object's ETag is remembered,
#   and an unchanged object is not downloaded again.
# - Writer mode (main pushes): <dir> must start empty. A runner-local cache
#   could hold files a pull-request job wrote, and an upload would publish
#   them under the trusted prefix; the writer instead compiles on top of
#   nothing but the trusted tarball.
#
# A writer uploads its whole directory as a single object. One PUT replaces
# the object atomically, so two concurrent main jobs cannot interleave one
# module's meta.json and objects: the later upload wins, and what it lacks is
# compiled and uploaded again by a later run. Nothing is uploaded when the
# job added nothing. When the tarball exceeds CI_COMPILE_CACHE_S3_MAX_MB
# (default 4096), only what this job compiled is uploaded, which resets the
# cache to the current working set: kernels of earlier FlashInfer versions
# and changed sources otherwise stay in it until the CI image changes.
#
# Only content-keyed artifacts are packed: CuTe-DSL modules (*_cute_dsl),
# TIRx (kda_tirx_*), KDA PTX (kda_ptx), downloaded cubins and the Triton
# cache. Never nvcc modules, tuning results or generated sources.
#
# Every failure only warns: the job then continues from the cache it has.

set -uo pipefail

MANIFEST=.s3-manifest
ETAG=.s3-etag
AWS_CLI_IMAGE=amazon/aws-cli:2.17.0

warn() {
    echo "::warning title=Compile cache S3::$*"
}

# The same sanitization ci/bash.sh applies to its per-image directory.
image_key() {
    printf '%s' "$1" | tr -c 'A-Za-z0-9._-' '_'
}

# Containers write the cache as root.
ensure_owned() {
    local dir=$1
    if [ -n "$(find "${dir}" ! -user "$(id -u)" -print -quit 2>/dev/null)" ]; then
        sudo -n chown -R "$(id -u):$(id -g)" "${dir}" 2>/dev/null \
            || { warn "cannot take ownership of ${dir}"; return 1; }
    fi
}

# NUL-separated paths of the cacheable units below the current directory.
list_units() {
    find . \( -path '*/cached_ops/*_cute_dsl' -o -path '*/cached_ops/kda_tirx_*' \
        -o -path '*/cached_ops/kda_ptx' -o -path './flashinfer/.cache/flashinfer/cubins' \
        -o -path ./triton \) -type d -prune -printf '%P\0' 2>/dev/null
}

# NUL-separated files of the given units, without in-flight temporaries;
# with NEWER set, only files modified after that file.
list_files() {
    local newer=()
    [ "$#" -gt 0 ] || return 0
    [ -z "${NEWER:-}" ] || newer=(-newer "${NEWER}")
    find "$@" -type f ! -name '*.lock' ! -name '*.tmp.*' ! -path '*/tmp.pid_*' \
        "${newer[@]}" -print0 2>/dev/null
}

# Fingerprint of the unit files of <dir>, to skip uploads that add nothing.
manifest() {
    (
        cd "$1" || exit 0
        mapfile -d '' -t units < <(list_units)
        NEWER="" list_files "${units[@]}" | xargs -0r stat -c '%n %s' | LC_ALL=C sort | sha256sum
    ) | cut -d' ' -f1
}

merge_tree() {
    local src=$1 dest=$2 rel
    while IFS= read -r -d '' rel; do
        if [ ! -e "${dest}/${rel}" ]; then
            mkdir -p "$(dirname "${dest}/${rel}")" && mv "${src}/${rel}" "${dest}/${rel}"
        elif cmp -s "${src}/${rel}/meta.json" "${dest}/${rel}/meta.json"; then
            # -n keeps every existing file; its exit status differs between
            # coreutils versions when something is skipped.
            cp -Rn --preserve=timestamps "${src}/${rel}/." "${dest}/${rel}/" 2>/dev/null
        fi
        rm -rf "${src:?}/${rel}"
    done < <(cd "${src}" && find . -path '*/cached_ops/*_cute_dsl' -type d -prune -print0)
    cp -Rn --preserve=timestamps "${src}/." "${dest}/" 2>/dev/null
    return 0
}

restore() {
    local base=$1 image=$2 lane=$3 dest=$4 mode=${5:-}
    local url="${base%/}/$(image_key "${image}")/${lane}.tar.gz"
    local tmp code etag headers=()
    mkdir -p "${dest}" && dest=$(cd "${dest}" && pwd) || { warn "cannot create ${dest}"; return 0; }
    if [ "${mode}" = --writer ]; then
        if [ -n "$(ls -A "${dest}")" ]; then
            warn "writer cache ${dest} is not empty; not restoring"
            return 0
        fi
        # Marks the start of the job for upload, also when nothing is restored.
        manifest "${dest}" > "${dest}/${MANIFEST}"
    elif [ -s "${dest}/${ETAG}" ]; then
        headers=(-H "If-None-Match: $(cat "${dest}/${ETAG}")")
    fi
    # A sibling of dest, so merged units are moved within one volume.
    tmp=$(mktemp -d "${dest%/}.s3-restore.XXXXXX") || { warn "no scratch space"; return 0; }
    code=$(curl -sSL --retry 3 --connect-timeout 10 --max-time 1800 "${headers[@]}" \
        -D "${tmp}/headers" -o "${tmp}/cache.tar.gz" -w '%{http_code}' "${url}")
    case "${code}" in
        200) ;;
        304)
            echo "Shared compile cache unchanged since this runner restored it: ${url}"
            rm -rf "${tmp}"
            return 0
            ;;
        403 | 404)
            echo "No shared compile cache at ${url}"
            rm -rf "${tmp}"
            return 0
            ;;
        *)
            warn "download of ${url} failed (HTTP ${code:-none}); continuing without it"
            rm -rf "${tmp}"
            return 0
            ;;
    esac
    etag=$(tr -d '\r' < "${tmp}/headers" | awk 'tolower($1) == "etag:" { e = $2 } END { print e }')
    if [ "${mode}" = --writer ]; then
        if tar -xzf "${tmp}/cache.tar.gz" -C "${dest}" --no-same-owner; then
            manifest "${dest}" > "${dest}/${MANIFEST}"
            echo "Restored shared compile cache ${url} ($(du -sm "${tmp}/cache.tar.gz" | cut -f1) MB)"
        else
            warn "cannot unpack ${url}; starting cold"
            find "${dest}" -mindepth 1 ! -name "${MANIFEST}" -delete
        fi
    elif ensure_owned "${dest}"; then
        mkdir "${tmp}/tree"
        if tar -xzf "${tmp}/cache.tar.gz" -C "${tmp}/tree" --no-same-owner; then
            merge_tree "${tmp}/tree" "${dest}"
            [ -z "${etag}" ] || printf '%s\n' "${etag}" > "${dest}/${ETAG}"
            echo "Merged shared compile cache ${url} ($(du -sm "${tmp}/cache.tar.gz" | cut -f1) MB)"
        else
            warn "cannot unpack ${url}; continuing without it"
        fi
    fi
    rm -rf "${tmp}"
}

# pack <dir> <tarball> <unit>...: the unit files (see list_files; paths are
# relative to <dir>), plus the meta.json of every CuTe-DSL module that
# contributes one.
pack() {
    local src=$1 out=$2
    shift 2
    if ! (
        cd "${src}" || exit 1
        list_files "$@" > "${out}.files"
        tr '\0' '\n' < "${out}.files" \
            | sed -n 's|^\(.*/cached_ops/[^/]*_cute_dsl\)/.*|\1/meta.json|p' \
            | LC_ALL=C sort -u | while IFS= read -r meta; do
                [ ! -f "${meta}" ] || printf '%s\0' "${meta}"
            done >> "${out}.files"
        LC_ALL=C sort -zu -o "${out}.files" "${out}.files"
        tar -czf "${out}" --null -T "${out}.files"
    ); then
        warn "cannot pack ${src}"
        rm -f "${out}.files"
        return 1
    fi
    rm -f "${out}.files"
}

s3_cp() {
    local dir=$1 file=$2 uri=$3
    if command -v aws > /dev/null; then
        aws s3 cp --only-show-errors "${dir}/${file}" "${uri}"
    else
        docker run --rm -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY \
            -e AWS_SESSION_TOKEN -e AWS_REGION -e AWS_DEFAULT_REGION \
            -v "${dir}:/upload:ro" "${AWS_CLI_IMAGE}" \
            s3 cp --only-show-errors "/upload/${file}" "${uri}"
    fi
}

upload() {
    local base=$1 image=$2 lane=$3 src=$4
    local uri="${base%/}/$(image_key "${image}")/${lane}.tar.gz"
    local max_mb=${CI_COMPILE_CACHE_S3_MAX_MB:-4096}
    local tmp size_mb units=()
    [ -d "${src}" ] || { echo "No compile cache at ${src}; nothing to upload"; return 0; }
    src=$(cd "${src}" && pwd) || return 0
    if [ -z "${AWS_ACCESS_KEY_ID:-}" ]; then
        warn "no AWS credentials; not uploading ${uri}"
        return 0
    fi
    ensure_owned "${src}" || return 0
    mapfile -d '' -t units < <(cd "${src}" && list_units)
    if [ "${#units[@]}" -eq 0 ]; then
        echo "No cacheable artifacts in ${src}; nothing to upload"
        return 0
    fi
    if [ -f "${src}/${MANIFEST}" ] && [ "$(manifest "${src}")" = "$(cat "${src}/${MANIFEST}")" ]; then
        echo "This job compiled nothing new; ${uri} is current"
        return 0
    fi
    tmp=$(mktemp -d "${src%/}.s3-upload.XXXXXX") || { warn "no scratch space"; return 0; }
    pack "${src}" "${tmp}/cache.tar.gz" "${units[@]}" || { rm -rf "${tmp}"; return 0; }
    size_mb=$(( $(stat -c %s "${tmp}/cache.tar.gz") / 1048576 ))
    if [ "${size_mb}" -gt "${max_mb}" ] && [ -f "${src}/${MANIFEST}" ]; then
        echo "Compile cache tarball is ${size_mb} MB, over ${max_mb} MB; uploading only this job's builds"
        NEWER="${MANIFEST}" pack "${src}" "${tmp}/cache.tar.gz" "${units[@]}" \
            || { rm -rf "${tmp}"; return 0; }
        size_mb=$(( $(stat -c %s "${tmp}/cache.tar.gz") / 1048576 ))
    fi
    if [ "${size_mb}" -gt "${max_mb}" ]; then
        warn "compile cache tarball is ${size_mb} MB, over ${max_mb} MB; not uploading"
    elif s3_cp "${tmp}" cache.tar.gz "${uri}"; then
        echo "Uploaded ${size_mb} MB to ${uri}"
    else
        warn "upload to ${uri} failed"
    fi
    rm -rf "${tmp}"
}

if [ "$#" -lt 5 ] || ! [[ $1 =~ ^(restore|upload)$ ]]; then
    echo "Usage: $0 restore <https-base> <image> <lane> <dir> [--writer]" >&2
    echo "       $0 upload <s3-base> <image> <lane> <dir>" >&2
    exit 2
fi
if ! [[ $4 =~ ^[A-Za-z0-9._-]+(/[A-Za-z0-9._-]+)*$ ]] || [[ $4 =~ (^|/)\.\.?(/|$) ]]; then
    warn "invalid lane '$4'; skipping"
    exit 0
fi
"$@"
exit 0
