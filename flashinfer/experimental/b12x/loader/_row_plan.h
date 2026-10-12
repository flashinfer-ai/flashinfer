/* Shared immutable row-source validation and allocation-free block planning. */
#ifndef B12X_ROW_PLAN_H
#define B12X_ROW_PLAN_H
#include <fcntl.h>
#include <time.h>
#define PLE_BLOCK 4096u   /* filesystem block size: the conservative fragment floor */
#define PLE_GRAN_MIN 512u /* NVMe logical block: the smallest fragment floor allowed */
#define PLE_READ_MAX 65536u
#define PLE_SORT_MIN 1024u /* waves below the shipped heapsort boundary always sort */

typedef struct {
  int fd;
  dev_t device;
  ino_t inode;
  int64_t bytes;
  unsigned gran; /* per-file fragment floor: PLE_BLOCK, or the DIO read alignment */
} ple_file_t;

typedef struct {
  int64_t shard;
  int64_t offset;
  size_t file;
  unsigned gran; /* the file's fragment floor, cached here for the hot planner */
  bool scale;
} ple_source_t;

typedef struct {
  size_t file;
  int64_t offset;
  unsigned length;
  size_t destination;
  bool scale;
} ple_fragment_t;

typedef struct {
  size_t file, begin, end;
  int64_t offset;
  unsigned length, expected;
} ple_job_t;

typedef struct {
  int64_t shard_rows, padded_rows, tp_start, tp_end;
  size_t weight_bytes, scale_bytes, max_lookups, capacity;
  ple_fragment_t* fragments;
  ple_job_t* jobs;
  size_t fragment_count, job_count;
  ple_file_t* files;
  ple_source_t* sources;
  size_t file_count, source_count;
  failure_t failure;
  uint64_t lookups, requested_bytes, read_bytes, read_calls;
  uint64_t unique_blocks, coalesced_reads, submit_calls;
  double execution_seconds, planning_seconds;
  bool io_uring_pick;  /* the miss-IO pick: DIOALIGN fragments + density-gated sort */
  unsigned min_gran;   /* smallest per-file fragment floor in use; 0 with no files */
  size_t* file_counts; /* per-wave fragment tally, sized file_counts_capacity */
  size_t file_counts_capacity;
} ple_plan_t;

/* Direct-I/O read alignment of an already-open source.
 *
 * iomap DIO checks reads against the device's *logical* block size only:
 * fs/iomap/direct-io.c (7.2 @ dd3210c4, iomap_dio_bio_iter) selects
 * `alignment = bdev_logical_block_size(bdev)` and requires
 * `(pos | length) & (alignment - 1) == 0`, keeping the stricter
 * IOMAP_DIO_FSBLOCK_ALIGNED for the write side of allocate-on-write
 * filesystems. statx() publishes that number per file: ext4 answers with
 * bdev_logical_block_size() (fs/ext4/inode.c:6214-6227), and XFS puts it in a
 * dedicated read field while keeping the allocation-unit hint in the write
 * field (fs/xfs/xfs_iops.c:582-599). A read path must therefore prefer
 * STATX_DIO_READ_ALIGN and only then fall back to STATX_DIOALIGN — which is
 * how the kernel's own DIO consumers treat the pair
 * (fs/nfsd/filecache.c:1117-1124, and a zero read alignment means no DIO at
 * all, fs/nfsd/vfs.c:1209).
 *
 * Anything the probe cannot answer — no statx, a filesystem that leaves both
 * mask bits unset (btrfs and tmpfs do here, measured on 7.2), an older
 * glibc/kernel, WSL2, or a device that reports more than 4 KiB — keeps the
 * historical PLE_BLOCK split, which is also the reference behaviour of this
 * header and can never be worse than what the reader did before. */
static inline unsigned ple_usable_align(unsigned candidate) {
  if (candidate >= PLE_GRAN_MIN && candidate <= PLE_BLOCK && (candidate & (candidate - 1u)) == 0)
    return candidate;
  return 0;
}

/* One self-contained guard: no intermediate feature macro to lose. glibc only
 * exposes struct statx (and these fields) under _GNU_SOURCE, which both
 * consumers of this header define before including it; the version floor is
 * where glibc synced the stx_dio_* fields in. */
#if defined(__linux__) && defined(__GLIBC__) && defined(STATX_DIOALIGN) && \
    (__GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ >= 34))
static inline unsigned ple_dio_gran(int fd) {
  struct statx probe;
  unsigned want = STATX_DIOALIGN;
#if defined(STATX_DIO_READ_ALIGN)
  want |= STATX_DIO_READ_ALIGN;
#endif
  if (statx(fd, "", AT_EMPTY_PATH, want, &probe) != 0) return PLE_BLOCK;
#if defined(STATX_DIO_READ_ALIGN)
  if (probe.stx_mask & STATX_DIO_READ_ALIGN) {
    unsigned align = ple_usable_align((unsigned)probe.stx_dio_read_offset_align);
    if (align) return align;
  }
#endif
  if (probe.stx_mask & STATX_DIOALIGN) {
    unsigned align = ple_usable_align((unsigned)probe.stx_dio_offset_align);
    if (align) return align;
  }
  return PLE_BLOCK;
}
#else
static inline unsigned ple_dio_gran(int fd) {
  (void)fd;
  return PLE_BLOCK;
}
#endif

static inline int ple_source_compare(const void* left, const void* right) {
  const ple_source_t *a = left, *b = right;
  if (a->shard != b->shard) return a->shard < b->shard ? -1 : 1;
  return (int)a->scale - (int)b->scale;
}

static inline void ple_plan_release(ple_plan_t* reader) {
  for (size_t i = 0; i < reader->file_count; i++) close(reader->files[i].fd);
  free(reader->files);
  free(reader->sources);
  free(reader->fragments);
  free(reader->jobs);
  free(reader->file_counts);
  reader->file_counts = NULL;
  reader->file_counts_capacity = 0;
}

static inline int ple_plan_init(ple_plan_t* reader, int64_t shard_rows, int64_t padded_rows,
                                int64_t tp_start, int64_t tp_end, int64_t weight_bytes,
                                int64_t scale_bytes, int64_t max_lookups) {
  if (shard_rows <= 0 || padded_rows <= 0 || tp_start < 0 || tp_end < tp_start ||
      tp_end > padded_rows || weight_bytes <= 0 || scale_bytes < 0 || max_lookups <= 0 ||
      weight_bytes > PY_SSIZE_T_MAX / max_lookups || scale_bytes > PY_SSIZE_T_MAX / max_lookups ||
      max_lookups > PY_SSIZE_T_MAX / 8)
    return EINVAL;
  size_t per_row = (size_t)weight_bytes / PLE_BLOCK + 2;
  if (scale_bytes) per_row += (size_t)scale_bytes / PLE_BLOCK + 2;
  if ((size_t)max_lookups > SIZE_MAX / per_row ||
      (size_t)max_lookups * per_row > SIZE_MAX / sizeof(ple_fragment_t) ||
      (size_t)max_lookups * per_row > SIZE_MAX / sizeof(ple_job_t))
    return EOVERFLOW;
  reader->shard_rows = shard_rows;
  reader->padded_rows = padded_rows;
  reader->tp_start = tp_start;
  reader->tp_end = tp_end;
  reader->weight_bytes = weight_bytes;
  reader->scale_bytes = scale_bytes;
  reader->max_lookups = max_lookups;
  reader->capacity = (size_t)max_lookups * per_row;
  /* `per_row` is bounded by PLE_BLOCK above, which is the only granularity
   * this initialiser may ever plan with: ple_add_source() keeps every file at
   * PLE_BLOCK unless the caller sets io_uring_pick first, and no caller that
   * sets it goes through this function (the io_uring reader sizes its own
   * capacity with PLE_GRAN_MIN). */
  reader->min_gran = 0;
  reader->file_counts = NULL;
  reader->file_counts_capacity = 0;
  reader->fragments = calloc(reader->capacity, sizeof(*reader->fragments));
  reader->jobs = calloc(reader->capacity, sizeof(*reader->jobs));
  return reader->fragments && reader->jobs ? 0 : ENOMEM;
}

static inline int ple_add_source(ple_plan_t* reader, int64_t shard, const char* path,
                                 int64_t offset, bool scale,
                                 int (*register_files)(ple_plan_t*, void*), void* owner) {
  if (shard < 0 || shard > (reader->padded_rows - 1) / reader->shard_rows || offset < 0 ||
      (scale && !reader->scale_bytes))
    return EINVAL;
  int64_t first = shard * reader->shard_rows;
  int64_t rows = reader->padded_rows - first;
  if (rows > reader->shard_rows) rows = reader->shard_rows;
  size_t row_bytes = scale ? reader->scale_bytes : reader->weight_bytes;
  if (rows > (INT64_MAX - offset) / (int64_t)row_bytes) return EOVERFLOW;
  if (first >= reader->tp_end || first + rows <= reader->tp_start) return 0;
  failure_t failure = {{0}};
  ple_source_t key = {.shard = shard, .scale = scale};
  if (reader->source_count &&
      bsearch(&key, reader->sources, reader->source_count, sizeof(key), ple_source_compare)) {
    snprintf(failure.message, sizeof(failure.message),
             "PLE source shard/plane is already registered");
  } else {
    /* Reject FIFOs via fstat without blocking waiting for a writer. */
    int fd = open(path, O_RDONLY | O_CLOEXEC | O_NONBLOCK | O_DIRECT);
    struct stat status;
    if (fd < 0)
      system_error(&failure, "open PLE source");
    else if (fstat(fd, &status) != 0)
      system_error(&failure, "fstat PLE source");
    else if (!S_ISREG(status.st_mode) || offset + rows * (int64_t)row_bytes > status.st_size)
      snprintf(failure.message, sizeof(failure.message),
               "PLE source range exceeds a regular file's size");
    else {
      size_t file = 0;
      while (file < reader->file_count && (reader->files[file].device != status.st_dev ||
                                           reader->files[file].inode != status.st_ino))
        file++;
      ple_source_t* sources =
          realloc(reader->sources, (reader->source_count + 1) * sizeof(*sources));
      if (!sources)
        snprintf(failure.message, sizeof(failure.message),
                 "could not allocate PLE source descriptor");
      else {
        reader->sources = sources;
        if (file == reader->file_count) {
          ple_file_t* files = realloc(reader->files, (file + 1) * sizeof(*files));
          if (!files)
            snprintf(failure.message, sizeof(failure.message),
                     "could not allocate PLE file descriptor");
          else {
            unsigned gran = reader->io_uring_pick ? ple_dio_gran(fd) : PLE_BLOCK;
            reader->files = files;
            reader->files[file] =
                (ple_file_t){fd, status.st_dev, status.st_ino, status.st_size, gran};
            reader->file_count++;
            /* The per-wave fragment tally grows with the file table rather
             * than with a wave, so metadata_bytes stays constant across runs
             * of a reader that has finished registering its sources. */
            if (reader->io_uring_pick && reader->file_count > reader->file_counts_capacity) {
              size_t* counts = realloc(reader->file_counts, reader->file_count * sizeof(*counts));
              if (!counts)
                snprintf(failure.message, sizeof(failure.message),
                         "could not allocate PLE fragment tally");
              else {
                reader->file_counts = counts;
                reader->file_counts_capacity = reader->file_count;
                reader->file_counts[reader->file_count - 1] = 0;
              }
            }
            if (!reader->min_gran || gran < reader->min_gran) reader->min_gran = gran;
            fd = -1;
            int error = register_files ? register_files(reader, owner) : 0;
            if (error < 0) {
              snprintf(failure.message, sizeof(failure.message), "row file registration failed: %s",
                       strerror(-error));
            }
          }
        } else if (reader->files[file].bytes != status.st_size) {
          snprintf(failure.message, sizeof(failure.message),
                   "PLE source file size changed during registration");
        }
        if (!failure.message[0]) {
          /* The fragment floor is copied out of the file table once, so the
           * per-row planner never has to chase a second pointer: sources are
           * the record the hot loop already holds. */
          reader->sources[reader->source_count++] = (ple_source_t){.shard = shard,
                                                                   .offset = offset,
                                                                   .file = file,
                                                                   .gran = reader->files[file].gran,
                                                                   .scale = scale != 0};
          qsort(reader->sources, reader->source_count, sizeof(*sources), ple_source_compare);
        }
      }
    }
    if (fd >= 0) close(fd);
  }
  if (failure.message[0] && !reader->failure.message[0]) reader->failure = failure;
  return failure.message[0] ? EIO : 0;
}

static inline int ple_fragment_compare(const void* left, const void* right) {
  const ple_fragment_t *a = left, *b = right;
  if (a->file != b->file) return a->file < b->file ? -1 : 1;
  return a->offset < b->offset ? -1 : a->offset > b->offset;
}

/* libc qsort may allocate a merge buffer on every run. Heap-sort in place to
 * keep all batch metadata persistent, including for duplicate-heavy batches. */
static inline void ple_sift(ple_fragment_t* items, size_t root, size_t count) {
  ple_fragment_t value = items[root];
  while (root < count / 2) {
    size_t child = root * 2 + 1;
    if (child + 1 < count && ple_fragment_compare(&items[child], &items[child + 1]) < 0) child++;
    if (ple_fragment_compare(&value, &items[child]) >= 0) break;
    items[root] = items[child];
    root = child;
  }
  items[root] = value;
}

static inline void ple_sort(ple_plan_t* reader) {
  ple_fragment_t* items = reader->fragments;
  size_t count = reader->fragment_count;
  if (count < 1024) {
    for (size_t i = count / 2; i > 0; i--) ple_sift(items, i - 1, count);
    for (size_t end = count; end > 1; end--) {
      ple_fragment_t value = items[end - 1];
      items[end - 1] = items[0];
      items[0] = value;
      ple_sift(items, 0, end - 1);
    }
    return;
  }
  /* Jobs are built only after sorting. Reuse that fixed-capacity allocation
   * as radix scratch instead of allocating another prefill-sized buffer. */
  _Static_assert(sizeof(ple_job_t) >= sizeof(ple_fragment_t),
                 "job allocation must cover fragment sort scratch");
  ple_fragment_t* scratch = (ple_fragment_t*)reader->jobs;
  ple_fragment_t *source = items, *target = scratch;
  uint64_t offset_variation = 0, file_variation = 0;
  for (size_t i = 1; i < count; i++) {
    offset_variation |= (uint64_t)(items[i].offset ^ items[0].offset);
    file_variation |= (uint64_t)(items[i].file ^ items[0].file);
  }
  /* Stable LSD passes sort offset first, then the primary file key. Skip
   * constant bytes, including the file key for a single-container table. */
  for (int file = 0; file < 2; file++) {
    uint64_t variation = file ? file_variation : offset_variation;
    for (unsigned shift = 0; shift < 64; shift += 8) {
      if (((variation >> shift) & 255) == 0) continue;
      size_t positions[256] = {0};
      for (size_t i = 0; i < count; i++) {
        uint64_t key = file ? (uint64_t)source[i].file : (uint64_t)source[i].offset;
        positions[(key >> shift) & 255]++;
      }
      size_t prefix = 0;
      for (unsigned i = 0; i < 256; i++) {
        size_t size = positions[i];
        positions[i] = prefix;
        prefix += size;
      }
      for (size_t i = 0; i < count; i++) {
        uint64_t key = file ? (uint64_t)source[i].file : (uint64_t)source[i].offset;
        target[positions[(key >> shift) & 255]++] = source[i];
      }
      ple_fragment_t* swap = source;
      source = target;
      target = swap;
    }
  }
  if (source != items) memcpy(items, source, count * sizeof(*items));
}

static inline bool ple_plan(ple_plan_t* reader, const char* ids, char* weights, char* scales,
                            size_t count) {
  reader->fragment_count = reader->job_count = 0;
  if (reader->io_uring_pick && reader->file_count) {
    if (reader->file_count > reader->file_counts_capacity) {
      size_t* counts = realloc(reader->file_counts, reader->file_count * sizeof(*counts));
      if (!counts) {
        snprintf(reader->failure.message, sizeof(reader->failure.message),
                 "could not allocate PLE fragment tally");
        return false;
      }
      reader->file_counts = counts;
      reader->file_counts_capacity = reader->file_count;
    }
    memset(reader->file_counts, 0, reader->file_count * sizeof(*reader->file_counts));
  }
  for (size_t i = 0; i < count; i++) {
    int64_t id;
    memcpy(&id, ids + i * sizeof(id), sizeof(id));
    if (id < reader->tp_start || id >= reader->tp_end) {
      if (weights) memset(weights + i * reader->weight_bytes, 0, reader->weight_bytes);
      if (scales && reader->scale_bytes)
        memset(scales + i * reader->scale_bytes, 0, reader->scale_bytes);
      continue;
    }
    for (int plane = 0; plane < (reader->scale_bytes ? 2 : 1); plane++) {
      ple_source_t key = {.shard = id / reader->shard_rows, .scale = plane != 0};
      ple_source_t* source = reader->source_count
                                 ? bsearch(&key, reader->sources, reader->source_count, sizeof(key),
                                           ple_source_compare)
                                 : NULL;
      if (!source) {
        snprintf(reader->failure.message, sizeof(reader->failure.message),
                 "missing PLE %s source for shard %lld", plane ? "scale" : "weight",
                 (long long)key.shard);
        return false;
      }
      size_t bytes = plane ? reader->scale_bytes : reader->weight_bytes;
      size_t destination = i * bytes;
      int64_t offset = source->offset + (id % reader->shard_rows) * (int64_t)bytes;
      reader->requested_bytes += bytes;
      /* One fragment per granularity-sized block, so a job never spans more
       * than its own aligned run; the floor is the file's DIO read alignment. */
      unsigned gran = source->gran;
      while (bytes) {
        unsigned length = gran - (uint64_t)offset % gran;
        if (bytes < length) length = (unsigned)bytes;
        reader->fragments[reader->fragment_count++] =
            (ple_fragment_t){source->file, offset, length, destination, plane != 0};
        if (reader->io_uring_pick) reader->file_counts[source->file]++;
        offset += length;
        destination += length;
        bytes -= length;
      }
    }
  }
  /* The sort exists to make adjacent blocks mergeable. On random row ids the
   * per-file fragment tally decides whether that can pay at all: two fragments
   * of one file are neighbours in `gran` units with probability ~2/sectors, so
   * a file holding `m` fragments of `sectors` positions has m(m-1)/sectors
   * expected merges (miss_io.md §1: 0.2 merges for a 1024-row wave spread over
   * 128 shards, ~1.5k for a 96,304-row wave). Skip the sort only when no file
   * reaches even one expected merge, and never below PLE_SORT_MIN fragments —
   * the shipped heapsort boundary, where the in-place sort costs a fraction of
   * a millisecond and a contiguous id run is a cheap win the wave would
   * otherwise lose. Byte results are identical either way: the jobs cover the
   * same fragments, only fewer of them merge. */
  bool skip_sort = reader->io_uring_pick && reader->fragment_count >= PLE_SORT_MIN;
  if (skip_sort) {
    for (size_t file = 0; file < reader->file_count; file++) {
      uint64_t tally = reader->file_counts[file];
      if (tally < 2) continue;
      uint64_t sectors = (uint64_t)reader->files[file].bytes / reader->files[file].gran + 1;
      if (tally * (tally - 1) >= sectors) {
        skip_sort = false;
        break;
      }
    }
  }
  if (!skip_sort) ple_sort(reader);
  size_t cursor = 0;
  while (cursor < reader->fragment_count) {
    size_t begin = cursor;
    ple_fragment_t* fragment = &reader->fragments[cursor];
    size_t file = fragment->file;
    unsigned gran = reader->files[file].gran;
    int64_t first = fragment->offset & ~(int64_t)(gran - 1);
    int64_t last = first;
    reader->unique_blocks++;
    cursor++;
    while (cursor < reader->fragment_count) {
      fragment = &reader->fragments[cursor];
      int64_t block = fragment->offset & ~(int64_t)(gran - 1);
      if (fragment->file != file ||
          (block != last && (block - last != (int64_t)gran || block - first >= PLE_READ_MAX)))
        break;
      if (block != last) {
        reader->unique_blocks++;
        last = block;
      }
      cursor++;
    }
    unsigned aligned = (unsigned)(last - first) + gran;
    int64_t available = reader->files[file].bytes - first;
    unsigned expected = available < aligned ? (unsigned)available : aligned;
    reader->jobs[reader->job_count++] = (ple_job_t){file, begin, cursor, first, aligned, expected};
    if (last != first) reader->coalesced_reads++;
  }
  return true;
}

#endif
