#!/usr/bin/env bash
# Shared S3 layer of the GPU CI compile cache (see ci/COMPILE_CACHE.md).
#
#   compile_cache_s3.sh fetch   <https-base> <image> <lane> <mirror>
#   compile_cache_s3.sh restore <https-base> <image> <lane> <dir>
#   compile_cache_s3.sh upload  <s3-base>    <image> <lane> <dir>
#   compile_cache_s3.sh merge   <src> <dest>
#
# The cache is one gzipped tarball per (CI image ID, GPU, lane), stored at
# <base>/<image>/<lane>.tar.gz under a prefix that only scheduled runs on main
# can write. Every job reads it over anonymous HTTPS.
#
# - fetch (pull request jobs, under ci/compile_cache.sh's runner lock): keep
#   <mirror>/tree an extracted copy of the tarball. The object's ETag is
#   remembered, so an unchanged tarball is not downloaded again. The tree is
#   mounted read-only; the job's entrypoint copies it into its private cache
#   and then merges the per-runner seed over it.
# - restore (writers): <dir> must be empty. The per-runner seed and mirror
#   live on a host where pull request jobs run, so a writer compiles on top of
#   nothing but the trusted tarball, and never publishes files those jobs
#   could have placed.
# - merge: add what <dest> lacks from <src>. A CuTe-DSL module directory is
#   one unit, because its single meta.json vouches for every object in it: a
#   module <dest> lacks is copied whole; one it has gets missing objects only
#   when both meta.json files are identical. Every other artifact is keyed by
#   its path, so missing files are copied and existing ones kept.
#
# A writer uploads its whole directory as a single object. One PUT replaces
# the object atomically, so two concurrent writers cannot interleave one
# module's meta.json and objects: the later upload wins, and what it lacks is
# compiled and uploaded again by a later run. Nothing is uploaded when the job
# added nothing, or when its restore failed: a writer that started cold by
# accident would replace the lane's tarball with only its own builds. When the
# tarball exceeds CI_COMPILE_CACHE_S3_MAX_MB (default 4096), only what this job
# compiled is uploaded. That bounds growth (kernels of earlier FlashInfer
# versions and changed sources otherwise stay until the CI image changes), but
# also drops what the job reused from the tarball; the next writers recompile
# and add those back. A lane whose working set exceeds the cap alternates
# between full and cut tarballs: raise the cap instead.
#
# Only content-keyed artifacts are packed: CuTe-DSL modules (*_cute_dsl),
# TIRx (kda_tirx_*), KDA PTX (kda_ptx), downloaded cubins and the Triton
# cache. Never nvcc modules, tuning results or generated sources.
#
# Every failure only warns: the job then continues from the cache it has.

set -uo pipefail

MANIFEST=.s3-manifest
AWS_CLI_IMAGE=amazon/aws-cli:2.17.0

warn() {
    echo "::warning title=Compile cache S3::$*"
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
# Contents, not sizes: a rebuilt module's meta.json keeps its size.
manifest() {
    (
        cd "$1" || exit 0
        mapfile -d '' -t units < <(list_units)
        NEWER="" list_files "${units[@]}" | xargs -0r sha256sum | LC_ALL=C sort -k2 | sha256sum
    ) | cut -d' ' -f1
}

# download <url> <file> [<etag>]: prints the HTTP status, and the response's
# ETag on a second line.
download() {
    local url=$1 out=$2 etag=${3:-} headers=()
    [ -z "${etag}" ] || headers=(-H "If-None-Match: ${etag}")
    curl -sSL --retry 3 --connect-timeout 10 --max-time 1800 "${headers[@]}" \
        -D "${out}.headers" -o "${out}" -w '%{http_code}\n' "${url}"
    tr -d '\r' < "${out}.headers" 2>/dev/null \
        | awk 'tolower($1) == "etag:" { e = $2 } END { print e }'
    rm -f "${out}.headers"
}

fetch() {
    local url="${1%/}/$2/$3.tar.gz" mirror=$4
    local tmp code etag old_etag=""
    mkdir -p "${mirror}" || { warn "cannot create ${mirror}"; return 0; }
    if [ -d "${mirror}/tree" ] && [ -s "${mirror}/etag" ]; then
        old_etag=$(cat "${mirror}/etag")
    fi
    tmp=$(mktemp -d "${mirror%/}/.fetch.XXXXXX") || { warn "no scratch space"; return 0; }
    { read -r code; read -r etag; } < <(download "${url}" "${tmp}/cache.tar.gz" "${old_etag}")
    case "${code}" in
        200)
            mkdir "${tmp}/tree"
            if tar -xzf "${tmp}/cache.tar.gz" -C "${tmp}/tree" --no-same-owner; then
                rm -rf "${mirror:?}/tree" "${mirror:?}/etag"
                mv "${tmp}/tree" "${mirror}/tree"
                [ -z "${etag}" ] || printf '%s\n' "${etag}" > "${mirror}/etag"
                echo "Fetched shared compile cache ${url} ($(du -sm "${tmp}/cache.tar.gz" | cut -f1) MB)"
            else
                warn "cannot unpack ${url}; continuing without it"
            fi
            ;;
        304) echo "Shared compile cache unchanged since this runner fetched it: ${url}" ;;
        403 | 404)
            echo "No shared compile cache at ${url}"
            rm -rf "${mirror:?}/tree" "${mirror:?}/etag"
            ;;
        *) warn "download of ${url} failed (HTTP ${code:-none}); continuing without it" ;;
    esac
    rm -rf "${tmp}"
}

restore() {
    local url="${1%/}/$2/$3.tar.gz" dest=$4
    local tmp code
    mkdir -p "${dest}" && dest=$(cd "${dest}" && pwd) || { warn "cannot create ${dest}"; return 0; }
    if [ -n "$(ls -A "${dest}")" ]; then
        warn "writer cache ${dest} is not empty; not restoring or uploading"
        return 0
    fi
    # The manifest marks the start of the job for upload, and its absence a
    # restore that did not complete: upload then skips.
    tmp=$(mktemp -d "${dest%/}.s3-restore.XXXXXX") || { warn "no scratch space"; return 0; }
    code=$(download "${url}" "${tmp}/cache.tar.gz" | head -n1)
    case "${code}" in
        200)
            if tar -xzf "${tmp}/cache.tar.gz" -C "${dest}" --no-same-owner; then
                manifest "${dest}" > "${dest}/${MANIFEST}"
                echo "Restored shared compile cache ${url} ($(du -sm "${tmp}/cache.tar.gz" | cut -f1) MB)"
            else
                warn "cannot unpack ${url}; starting cold, and this job will not upload"
                find "${dest}" -mindepth 1 -delete
            fi
            ;;
        404)
            # Only a 404 proves the tarball absent: S3 answers 403 both for a
            # denied read of an existing object and, without s3:ListBucket, for
            # a missing one, and uploading after either would replace the lane.
            echo "No shared compile cache at ${url}"
            manifest "${dest}" > "${dest}/${MANIFEST}"
            ;;
        *) warn "download of ${url} failed (HTTP ${code:-none}); starting cold, and this job will not upload" ;;
    esac
    rm -rf "${tmp}"
}

merge() {
    local src=$1 dest=$2 rel
    [ -d "${src}" ] || return 0
    while IFS= read -r -d '' rel; do
        if [ ! -e "${dest}/${rel}" ]; then
            mkdir -p "$(dirname "${dest}/${rel}")" && cp -a "${src}/${rel}" "${dest}/${rel}"
        elif cmp -s "${src}/${rel}/meta.json" "${dest}/${rel}/meta.json"; then
            # -n keeps every existing file; its exit status differs between
            # coreutils versions when something is skipped.
            cp -Rn --preserve=timestamps "${src}/${rel}/." "${dest}/${rel}/" 2>/dev/null
        fi
    done < <(cd "${src}" && find . -path '*/cached_ops/*_cute_dsl' -type d -prune -print0)
    tar -C "${src}" --exclude='*/cached_ops/*_cute_dsl' -cf - . \
        | tar -C "${dest}" --skip-old-files -xf - \
        || warn "cannot merge ${src} into ${dest}"
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
    local uri="${1%/}/$2/$3.tar.gz" src=$4
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
    if ! [ -f "${src}/${MANIFEST}" ]; then
        warn "the writer's restore did not complete; not uploading ${uri}"
        return 0
    fi
    if [ "$(manifest "${src}")" = "$(cat "${src}/${MANIFEST}")" ]; then
        echo "This job compiled nothing new; ${uri} is current"
        return 0
    fi
    tmp=$(mktemp -d "${src%/}.s3-upload.XXXXXX") || { warn "no scratch space"; return 0; }
    pack "${src}" "${tmp}/cache.tar.gz" "${units[@]}" || { rm -rf "${tmp}"; return 0; }
    size_mb=$(( $(stat -c %s "${tmp}/cache.tar.gz") / 1048576 ))
    if [ "${size_mb}" -gt "${max_mb}" ]; then
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

case "${1:-}:$#" in
    merge:3) ;;
    fetch:5 | restore:5 | upload:5)
        # Both become parts of an object key and a local path.
        if ! [[ $3 =~ ^[A-Za-z0-9._-]+$ ]] || [[ $3 =~ ^\.\.?$ ]] \
            || ! [[ $4 =~ ^[A-Za-z0-9._-]+(/[A-Za-z0-9._-]+)*$ ]] \
            || [[ $4 =~ (^|/)\.\.?(/|$) ]]; then
            warn "invalid image '$3' or lane '$4'; skipping"
            exit 0
        fi
        ;;
    *)
        echo "Usage: $0 fetch|restore <https-base> <image> <lane> <dir>" >&2
        echo "       $0 upload <s3-base> <image> <lane> <dir>" >&2
        echo "       $0 merge <src> <dest>" >&2
        exit 2
        ;;
esac
"$@"
exit 0
