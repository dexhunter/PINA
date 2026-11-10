/* Count malloc-family calls while osspr_reset() is armed.
 *
 * Python is started with PYTHONMALLOC=malloc so each Python object is its
 * own libc allocation. The counter sits in front of glibc and does not
 * allocate.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

static _Atomic size_t g_calls;
static _Atomic int g_enabled;
static _Atomic int g_ready;

static void *(*real_malloc)(size_t);
static void *(*real_calloc)(size_t, size_t);
static void *(*real_realloc)(void *, size_t);
static void *(*real_aligned_alloc)(size_t, size_t);
static int (*real_posix_memalign)(void **, size_t, size_t);
static void *(*real_memalign)(size_t, size_t);

static void init(void) {
    if (atomic_load(&g_ready)) {
        return;
    }
    real_malloc = dlsym(RTLD_NEXT, "malloc");
    real_calloc = dlsym(RTLD_NEXT, "calloc");
    real_realloc = dlsym(RTLD_NEXT, "realloc");
    real_aligned_alloc = dlsym(RTLD_NEXT, "aligned_alloc");
    real_posix_memalign = dlsym(RTLD_NEXT, "posix_memalign");
    real_memalign = dlsym(RTLD_NEXT, "memalign");
    atomic_store(&g_ready, 1);
}

__attribute__((constructor)) static void ctor(void) { init(); }

void osspr_reset(void) {
    atomic_store(&g_calls, 0);
    atomic_store(&g_enabled, 1);
}

void osspr_stop(void) { atomic_store(&g_enabled, 0); }

size_t osspr_count(void) { return atomic_load(&g_calls); }

static void bump(void) {
    if (atomic_load(&g_enabled)) {
        atomic_fetch_add(&g_calls, 1);
    }
}

void *malloc(size_t n) {
    if (!real_malloc) {
        init();
    }
    void *p = real_malloc(n);
    bump();
    return p;
}

void *calloc(size_t n, size_t s) {
    if (!real_calloc) {
        init();
    }
    void *p = real_calloc(n, s);
    bump();
    return p;
}

void *realloc(void *p, size_t n) {
    if (!real_realloc) {
        init();
    }
    void *q = real_realloc(p, n);
    bump();
    return q;
}

void *aligned_alloc(size_t a, size_t n) {
    if (!real_aligned_alloc) {
        init();
    }
    void *p = real_aligned_alloc(a, n);
    bump();
    return p;
}

int posix_memalign(void **out, size_t a, size_t n) {
    if (!real_posix_memalign) {
        init();
    }
    int rc = real_posix_memalign(out, a, n);
    if (rc == 0) {
        bump();
    }
    return rc;
}

void *memalign(size_t a, size_t n) {
    if (!real_memalign) {
        init();
    }
    void *p = real_memalign(a, n);
    bump();
    return p;
}
