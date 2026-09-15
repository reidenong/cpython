#ifdef _Py_TIER2

#include "Python.h"

#include "pycore_code.h"
#include "pycore_frame.h"
#include "pycore_interpframe.h"
#include "pycore_jit.h"
#include "pycore_long.h"
#include "pycore_opcode_metadata.h"  // HAS_PURE_FLAG
#include "pycore_optimizer.h"
#include "pycore_stats.h"
#include "pycore_uop_metadata.h"  // MAX_CACHED_REGISTER

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

/*

Symbols
=======

https://github.com/faster-cpython/ideas/blob/main/3.13/redundancy_eliminator.md

Logically, all symbols begin as UNKNOWN, and can transition downwards along the
edges of the lattice, but *never* upwards (see the diagram below). The UNKNOWN
state represents no information, and the BOTTOM state represents contradictory
information. Though symbols logically progress through all intermediate nodes,
we often skip in-between states for convenience:

   UNKNOWN---------------------+------+
   |     |                     |      |
NULL     |                     |      RECORDED_VALUE*
|        |                     |      |            <- Anything below this level is an object.
|        NON_NULL---------+    |      |
|          |              |    |      |            <- Anything below this level has a known type version.
|    TYPE_VERSION         |    |      |
|    |                    |    |      |            <- Anything below this level has a known type.
|    KNOWN_CLASS--+       |    |      |
|    |  |  |              | PREDICATE RECORDED_VALUE(known type)
|    |  | INT*            |    |      |
|    |  |  |              |    |      |            <- Anything below this level has a known truthiness.
|    |  |  |              |    |      |
| TUPLE |  |        TRUTHINESS |      |
|    |  |  |              |    |      |            <- Anything below this level is a known constant.
|    KNOWN_VALUE--+-------+----+------+
|    |                                             <- Anything below this level is unreachable.
BOTTOM



For example, after guarding that the type of an UNKNOWN local is int, we can
narrow the symbol to KNOWN_CLASS (logically progressing though NON_NULL and
TYPE_VERSION to get there). Later, we may learn that it is falsey based on the
result of a truth test, which would allow us to narrow the symbol to KNOWN_VALUE
(with a value of integer zero). If at any point we encounter a float guard on
the same symbol, that would be a contradiction, and the symbol would be set to
BOTTOM (indicating that the code is unreachable).

INT* is a limited range int, currently a "compact" int.
RECORDED_VALUE* includes RECORDED_TYPE and RECORDED_GEN_FUNC

*/

#ifdef Py_DEBUG
static inline int get_lltrace(void) {
    char *uop_debug = Py_GETENV("PYTHON_OPT_DEBUG");
    int lltrace = 0;
    if (uop_debug != NULL && *uop_debug >= '0') {
        lltrace = *uop_debug - '0';  // TODO: Parse an int and all that
    }
    return lltrace;
}
#define DPRINTF(level, ...) \
    if (get_lltrace() >= (level)) { printf(__VA_ARGS__); }

void
_PyUOpSymPrint(JitOptContext *ctx, JitOptRef ref)
{
    if (PyJitRef_IsNull(ref)) {
        printf("<JitRef NULL>");
        return;
    }
    if (PyJitRef_IsInvalid(ref)) {
        printf("<INVALID frame at %p>", (void *)PyJitRef_Unwrap(ref));
        return;
    }
    JitOptSymbol *sym = PyJitRef_Unwrap(ref);
    JitSymType tag = sym->tag;
    switch (tag) {
        case JIT_SYM_UNKNOWN_TAG:
            printf("<? at %p>", (void *)sym);
            break;
        case JIT_SYM_NULL_TAG:
            printf("<NULL at %p>", (void *)sym);
            break;
        case JIT_SYM_NON_NULL_TAG:
            printf("<!NULL at %p>", (void *)sym);
            break;
        case JIT_SYM_BOTTOM_TAG:
            printf("<BOTTOM at %p>", (void *)sym);
            break;
        case JIT_SYM_TYPE_VERSION_TAG:
            printf("<v%u at %p>", sym->version.version, (void *)sym);
            break;
        case JIT_SYM_KNOWN_CLASS_TAG:
            printf("<%s at %p>", sym->cls.type->tp_name, (void *)sym);
            break;
        case JIT_SYM_KNOWN_VALUE_TAG:
            printf("<%s val=%p at %p>", Py_TYPE(sym->value.value)->tp_name,
                   (void *)sym->value.value, (void *)sym);
            break;
        case JIT_SYM_TUPLE_TAG:
            printf("<tuple[%d] at %p>", sym->tuple.length, (void *)sym);
            break;
        case JIT_SYM_TRUTHINESS_TAG:
            printf("<truthiness%s at %p>", sym->truthiness.invert ? "!" : "", (void *)sym);
            break;
        case JIT_SYM_COMPACT_INT:
            printf("<compact_int at %p>", (void *)sym);
            break;
        case JIT_SYM_PREDICATE_TAG:
            printf("<predicate at %p>", (void *)sym);
            break;
        case JIT_SYM_RECORDED_VALUE_TAG:
            printf("<recorded value %p>", sym->recorded_value.value);
            break;
        case JIT_SYM_RECORDED_TYPE_TAG:
            printf("<recorded type %s>", sym->recorded_type.type->tp_name);
            break;
        case JIT_SYM_RECORDED_GEN_FUNC_TAG:
            printf("<recorded gen func at %p>", (void *)sym);
            break;
        default:
            printf("<tag=%d at %p>", tag, (void *)sym);
            break;
    }
    JitOptAllocation allocation = _Py_uop_alloc_get(ctx, ref);
    if (allocation.state == JIT_ALLOC_VIRTUAL) {
        const JitVirtualObject *virtual_object =
            _Py_uop_alloc_resolve_virtual(ctx, ref);
        if (virtual_object != NULL &&
            virtual_object->kind == JIT_VIRTUAL_FLOAT)
        {
            printf(" [virtual float #%u]",
                   (unsigned int)allocation.virtual_ref);
        }
        else {
            printf(" [invalid virtual #%u]",
                   (unsigned int)allocation.virtual_ref);
        }
    }
}

#else
#define DPRINTF(level, ...)
#endif


static JitOptSymbol NO_SPACE_SYMBOL = {
    .tag = JIT_SYM_BOTTOM_TAG
};

static JitOptSymbol *
allocation_base(JitOptContext *ctx)
{
    return ctx->t_arena.arena;
}

JitOptSymbol *
out_of_space(JitOptContext *ctx)
{
    ctx->done = true;
    ctx->out_of_space = true;
    return &NO_SPACE_SYMBOL;
}

JitOptRef
out_of_space_ref(JitOptContext *ctx)
{
    return PyJitRef_Wrap(out_of_space(ctx));
}

static JitOptSymbol *
sym_new(JitOptContext *ctx)
{
    if (ctx->t_arena.ty_curr_number >= ctx->t_arena.ty_max_number) {
        OPT_STAT_INC(optimizer_failure_reason_no_memory);
        DPRINTF(1, "out of space for symbolic expression type\n");
        return NULL;
    }
    int index = ctx->t_arena.ty_curr_number++;
    JitOptSymbol *self = &ctx->t_arena.arena[index];
    self->tag = JIT_SYM_UNKNOWN_TAG;
    /* Ordinary symbols are concrete unless a virtual-producing uop explicitly
     * changes their allocation representation. */
    ctx->allocations[index] = JIT_VIRTUAL_REF_NULL;
    return self;
}

/* Allocation representation
 * =========================
 *
 * This table is deliberately independent of JitOptSymbol's type/value
 * lattice.  Reference-ownership tags are stripped by PyJitRef_Unwrap(), so
 * borrowed and unique forms of one symbol share a table entry.
 */

static bool
allocation_index(JitOptContext *ctx, JitOptRef ref, int *index)
{
    if (PyJitRef_IsNull(ref) || PyJitRef_IsInvalid(ref)) {
        return false;
    }
    uintptr_t address = (uintptr_t)PyJitRef_Unwrap(ref);
    uintptr_t base = (uintptr_t)ctx->t_arena.arena;
    uintptr_t end = base + (size_t)ctx->t_arena.ty_curr_number * sizeof(JitOptSymbol);
    if (address < base || address >= end ||
        (address - base) % sizeof(JitOptSymbol) != 0)
    {
        return false;
    }
    *index = (int)((address - base) / sizeof(JitOptSymbol));
    return true;
}

JitOptAllocation
_Py_uop_alloc_get(JitOptContext *ctx, JitOptRef ref)
{
    int index;
    if (!allocation_index(ctx, ref, &index)) {
        /* Invalid optimizer references cannot represent an unboxed runtime
         * value, so read-only queries conservatively treat them as concrete. */
        return (JitOptAllocation){
            .state = JIT_ALLOC_CONCRETE,
            .virtual_ref = JIT_VIRTUAL_REF_NULL,
        };
    }
    JitVirtualRef virtual_ref = ctx->allocations[index];
    return (JitOptAllocation){
        .state = virtual_ref == JIT_VIRTUAL_REF_NULL
            ? JIT_ALLOC_CONCRETE : JIT_ALLOC_VIRTUAL,
        .virtual_ref = virtual_ref,
    };
}

bool
_Py_uop_alloc_is_virtual(JitOptContext *ctx, JitOptRef ref)
{
    return _Py_uop_alloc_get(ctx, ref).state == JIT_ALLOC_VIRTUAL;
}

bool
_Py_uop_alloc_set_concrete(JitOptContext *ctx, JitOptRef ref)
{
    /* This changes optimizer metadata only.  Once virtual uops have been
     * emitted, callers must materialize into a fresh symbol instead. */
    int index;
    if (!allocation_index(ctx, ref, &index)) {
        return false;
    }
    ctx->allocations[index] = JIT_VIRTUAL_REF_NULL;
    return true;
}

JitVirtualRef
_Py_uop_alloc_new_virtual_float(JitOptContext *ctx, JitOptRef ref)
{
    int allocation;
    if (!allocation_index(ctx, ref, &allocation) ||
        !_Py_uop_sym_matches_type(ref, &PyFloat_Type))
    {
        return JIT_VIRTUAL_REF_NULL;
    }
    JitVirtualRef existing = ctx->allocations[allocation];
    if (existing != JIT_VIRTUAL_REF_NULL) {
        return existing;
    }
    if (ctx->virtuals.virtual_curr_number >= ctx->virtuals.virtual_max_number) {
        OPT_STAT_INC(optimizer_failure_reason_no_memory);
        DPRINTF(1, "out of space for virtual-object descriptors\n");
        ctx->retry_without_virtuals = true;
        out_of_space(ctx);
        return JIT_VIRTUAL_REF_NULL;
    }
    int virtual_index = ctx->virtuals.virtual_curr_number++;
    ctx->virtuals.arena[virtual_index].kind = JIT_VIRTUAL_FLOAT;
    JitVirtualRef virtual_ref = (JitVirtualRef)(virtual_index + 1);
    ctx->allocations[allocation] = virtual_ref;
    return virtual_ref;
}

JitOptRef
_Py_uop_alloc_materialize_virtual_float(JitOptContext *ctx, JitOptRef ref)
{
    /* Materialization changes the runtime representation, so it produces a
     * fresh concrete symbol rather than mutating the virtual symbol and any
     * aliases which may eventually share it.  Stage 5 accepts only unique
     * values; alias-aware, once-per-identity materialization comes later. */
    if (!PyJitRef_IsUnique(ref)) {
        return PyJitRef_NULL;
    }
    const JitVirtualObject *virtual_object =
        _Py_uop_alloc_resolve_virtual(ctx, ref);
    if (virtual_object == NULL ||
        virtual_object->kind != JIT_VIRTUAL_FLOAT ||
        !_Py_uop_sym_matches_type(ref, &PyFloat_Type))
    {
        return PyJitRef_NULL;
    }
    JitOptRef concrete = PyJitRef_MakeUnique(
        _Py_uop_sym_new_type(ctx, &PyFloat_Type));
    assert(!_Py_uop_alloc_is_virtual(ctx, concrete));
    return concrete;
}

const JitVirtualObject *
_Py_uop_alloc_resolve_virtual(JitOptContext *ctx, JitOptRef ref)
{
    JitOptAllocation allocation = _Py_uop_alloc_get(ctx, ref);
    if (allocation.state != JIT_ALLOC_VIRTUAL) {
        return NULL;
    }
    int index = (int)allocation.virtual_ref - 1;
    if (index < 0 || index >= ctx->virtuals.virtual_curr_number) {
        return NULL;
    }
    return &ctx->virtuals.arena[index];
}

static bool
allocation_is_valid(JitOptContext *ctx, JitOptRef ref)
{
    JitOptAllocation allocation = _Py_uop_alloc_get(ctx, ref);
    if (allocation.state == JIT_ALLOC_CONCRETE) {
        return allocation.virtual_ref == JIT_VIRTUAL_REF_NULL;
    }
    const JitVirtualObject *virtual_object =
        _Py_uop_alloc_resolve_virtual(ctx, ref);
    return virtual_object != NULL &&
        virtual_object->kind == JIT_VIRTUAL_FLOAT &&
        _Py_uop_sym_matches_type(ref, &PyFloat_Type);
}

static bool
valid_frame_range(_Py_UOpsAbstractFrame *frame, JitOptRef *stack_pointer)
{
    if (frame == NULL || frame->locals == NULL || frame->stack == NULL) {
        return false;
    }
    return stack_pointer >= frame->stack &&
        stack_pointer <= frame->stack + frame->stack_len + MAX_CACHED_REGISTER;
}

bool
_Py_uop_alloc_frame_has_virtual(
    JitOptContext *ctx,
    _Py_UOpsAbstractFrame *frame,
    JitOptRef *stack_pointer)
{
    if (!valid_frame_range(frame, stack_pointer)) {
        /* A future safety barrier must not mistake malformed optimizer state
         * for proof that every live value is concrete. */
        return true;
    }
    for (int i = 0; i < frame->locals_len; i++) {
        if (_Py_uop_alloc_is_virtual(ctx, frame->locals[i])) {
            return true;
        }
    }
    for (JitOptRef *ref = frame->stack; ref < stack_pointer; ref++) {
        if (_Py_uop_alloc_is_virtual(ctx, *ref)) {
            return true;
        }
    }
    return false;
}

int
_Py_uop_alloc_frame_virtual_count(
    JitOptContext *ctx,
    _Py_UOpsAbstractFrame *frame,
    JitOptRef *stack_pointer)
{
    if (!valid_frame_range(frame, stack_pointer)) {
        return -1;
    }
    int count = 0;
    for (int i = 0; i < frame->locals_len; i++) {
        count += _Py_uop_alloc_is_virtual(ctx, frame->locals[i]);
    }
    for (JitOptRef *ref = frame->stack; ref < stack_pointer; ref++) {
        count += _Py_uop_alloc_is_virtual(ctx, *ref);
    }
    return count;
}

int
_Py_uop_alloc_context_virtual_count(
    JitOptContext *ctx,
    JitOptRef *current_stack_pointer)
{
    if (ctx->virtuals.virtual_curr_number == 0) {
        return 0;
    }
    int count = 0;
    for (int i = 0; i < ctx->curr_frame_depth; i++) {
        _Py_UOpsAbstractFrame *frame = &ctx->frames[i];
        JitOptRef *stack_pointer = frame == ctx->frame
            ? current_stack_pointer : frame->stack_pointer;
        int frame_count = _Py_uop_alloc_frame_virtual_count(
            ctx, frame, stack_pointer);
        if (frame_count < 0) {
            return -1;
        }
        count += frame_count;
    }
    return count;
}

bool
_Py_uop_alloc_context_has_virtual(
    JitOptContext *ctx,
    JitOptRef *current_stack_pointer)
{
    /* Invalid frame state is not proof that every runtime slot is boxed. */
    return _Py_uop_alloc_context_virtual_count(
        ctx, current_stack_pointer) != 0;
}

bool
_Py_uop_alloc_validate_frame(
    JitOptContext *ctx,
    _Py_UOpsAbstractFrame *frame,
    JitOptRef *stack_pointer)
{
    if (!valid_frame_range(frame, stack_pointer)) {
        return false;
    }
    for (int i = 0; i < frame->locals_len; i++) {
        if (!allocation_is_valid(ctx, frame->locals[i])) {
            return false;
        }
    }
    for (JitOptRef *ref = frame->stack; ref < stack_pointer; ref++) {
        if (!allocation_is_valid(ctx, *ref)) {
            return false;
        }
    }
    return true;
}

bool
_Py_uop_alloc_validate_context(
    JitOptContext *ctx,
    JitOptRef *current_stack_pointer)
{
    for (int i = 0; i < ctx->curr_frame_depth; i++) {
        _Py_UOpsAbstractFrame *frame = &ctx->frames[i];
        JitOptRef *stack_pointer = frame == ctx->frame
            ? current_stack_pointer : frame->stack_pointer;
        if (!_Py_uop_alloc_validate_frame(ctx, frame, stack_pointer)) {
            return false;
        }
    }
    return true;
}

static void make_const(JitOptSymbol *sym, PyObject *val)
{
    sym->tag = JIT_SYM_KNOWN_VALUE_TAG;
    sym->value.value = Py_NewRef(val);
}

static inline void
sym_set_bottom(JitOptContext *ctx, JitOptSymbol *sym)
{
    sym->tag = JIT_SYM_BOTTOM_TAG;
    ctx->done = true;
    ctx->contradiction = true;
}

bool
_Py_uop_sym_is_bottom(JitOptRef ref)
{
    JitOptSymbol *sym = PyJitRef_Unwrap(ref);
    return sym->tag == JIT_SYM_BOTTOM_TAG;
}

bool
_Py_uop_sym_is_not_null(JitOptRef ref) {
    JitOptSymbol *sym = PyJitRef_Unwrap(ref);
    return sym->tag == JIT_SYM_NON_NULL_TAG || sym->tag > JIT_SYM_BOTTOM_TAG;
}

bool
_Py_uop_sym_is_const(JitOptContext *ctx, JitOptRef ref)
{
    JitOptSymbol *sym = PyJitRef_Unwrap(ref);
    if (sym->tag == JIT_SYM_KNOWN_VALUE_TAG) {
        return true;
    }
    if (sym->tag == JIT_SYM_TRUTHINESS_TAG) {
        JitOptSymbol *value = allocation_base(ctx) + sym->truthiness.value;
        int truthiness = _Py_uop_sym_truthiness(ctx, PyJitRef_Wrap(value));
        if (truthiness < 0) {
            return false;
        }
        make_const(sym, (truthiness ^ sym->truthiness.invert) ? Py_True : Py_False);
        return true;
    }
    return false;
}

bool
_Py_uop_sym_is_null(JitOptRef ref)
{
    return PyJitRef_Unwrap(ref)->tag == JIT_SYM_NULL_TAG;
}


PyObject *
_Py_uop_sym_get_const(JitOptContext *ctx, JitOptRef ref)
{
    JitOptSymbol *sym = PyJitRef_Unwrap(ref);
    if (sym->tag == JIT_SYM_KNOWN_VALUE_TAG) {
        return sym->value.value;
    }
    if (sym->tag == JIT_SYM_TRUTHINESS_TAG) {
        JitOptSymbol *value = allocation_base(ctx) + sym->truthiness.value;
        int truthiness = _Py_uop_sym_truthiness(ctx, PyJitRef_Wrap(value));
        if (truthiness < 0) {
            return NULL;
        }
        PyObject *res = (truthiness ^ sym->truthiness.invert) ? Py_True : Py_False;
        make_const(sym, res);
        return res;
    }
    return NULL;
}

_PyStackRef
_Py_uop_sym_get_const_as_stackref(JitOptContext *ctx, JitOptRef sym)
{
    PyObject *const_val = _Py_uop_sym_get_const(ctx, sym);
    if (const_val == NULL) {
        return PyStackRef_NULL;
    }
    return PyStackRef_FromPyObjectBorrow(const_val);
}

/*
 Indicates whether the constant is safe to constant evaluate
 (without side effects).
 */
bool
_Py_uop_sym_is_safe_const(JitOptContext *ctx, JitOptRef sym)
{
    PyObject *const_val = _Py_uop_sym_get_const(ctx, sym);
    if (const_val == NULL) {
        return false;
    }
    if (_PyLong_CheckExactAndCompact(const_val)) {
        return true;
    }
    PyTypeObject *typ = Py_TYPE(const_val);
    return (typ == &PyUnicode_Type) ||
           (typ == &PyFloat_Type) ||
           (typ == &_PyNone_Type) ||
           (typ == &PyBool_Type) ||
           (typ == &PyFrozenDict_Type) ||
           (typ == &PyFrozenSet_Type);
}

bool
_Py_uop_sym_is_not_container(JitOptRef sym)
{
    PyTypeObject *typ = _Py_uop_sym_get_type(sym);
    if (typ == NULL) {
        return false;
    }
    return (typ == &PyLong_Type) ||
           (typ == &PyFloat_Type) ||
           (typ == &PyUnicode_Type) ||
           (typ == &_PyNone_Type) ||
           (typ == &PyBool_Type);
}

void
_Py_uop_sym_set_type(JitOptContext *ctx, JitOptRef ref, PyTypeObject *typ)
{
    JitOptSymbol *sym = PyJitRef_Unwrap(ref);
    JitSymType tag = sym->tag;
    switch(tag) {
        case JIT_SYM_NULL_TAG:
            sym_set_bottom(ctx, sym);
            return;
        case JIT_SYM_KNOWN_CLASS_TAG:
            if (sym->cls.type != typ) {
                sym_set_bottom(ctx, sym);
            }
            return;
        case JIT_SYM_TYPE_VERSION_TAG:
            if (sym->version.version == typ->tp_version_tag) {
                sym->tag = JIT_SYM_KNOWN_CLASS_TAG;
                sym->cls.type = typ;
                sym->cls.version = typ->tp_version_tag;
            }
            else {
                sym_set_bottom(ctx, sym);
            }
            return;
        case JIT_SYM_KNOWN_VALUE_TAG:
            if (Py_TYPE(sym->value.value) != typ) {
                Py_CLEAR(sym->value.value);
                sym_set_bottom(ctx, sym);
            }
            return;
        case JIT_SYM_TUPLE_TAG:
            if (typ != &PyTuple_Type) {
                sym_set_bottom(ctx, sym);
            }
            return;
        case JIT_SYM_RECORDED_GEN_FUNC_TAG:
            if (typ != &PyGen_Type) {
                sym_set_bottom(ctx, sym);
            }
            return;
        case JIT_SYM_BOTTOM_TAG:
            return;
        case JIT_SYM_RECORDED_VALUE_TAG:
            if (Py_TYPE(sym->recorded_value.value) == typ) {
                sym->recorded_value.known_type = true;
            }
            else {
                sym_set_bottom(ctx, sym);
            }
            return;
        case JIT_SYM_RECORDED_TYPE_TAG:
            if (sym->recorded_type.type == typ) {
                sym->tag = JIT_SYM_KNOWN_CLASS_TAG;
                sym->cls.version = 0;
                sym->cls.type = typ;
            }
            else {
                sym_set_bottom(ctx, sym);
            }
            return;
        case JIT_SYM_NON_NULL_TAG:
        case JIT_SYM_UNKNOWN_TAG:
            sym->tag = JIT_SYM_KNOWN_CLASS_TAG;
            sym->cls.version = 0;
            sym->cls.type = typ;
            return;
        case JIT_SYM_PREDICATE_TAG:
        case JIT_SYM_TRUTHINESS_TAG:
            if (typ != &PyBool_Type) {
                sym_set_bottom(ctx, sym);
            }
            return;
        case JIT_SYM_COMPACT_INT:
            if (typ != &PyLong_Type) {
                sym_set_bottom(ctx, sym);
            }
            return;
    }
}

bool
_Py_uop_sym_set_type_version(JitOptContext *ctx, JitOptRef ref, unsigned int version)
{
    JitOptSymbol *sym = PyJitRef_Unwrap(ref);
    PyTypeObject *type = _PyType_LookupByVersion(version);
    if (type) {
        _Py_uop_sym_set_type(ctx, ref, type);
    }
    JitSymType tag = sym->tag;
    switch(tag) {
        case JIT_SYM_NULL_TAG:
            sym_set_bottom(ctx, sym);
            return false;
        case JIT_SYM_KNOWN_CLASS_TAG:
            if (sym->cls.type->tp_version_tag != version) {
                sym_set_bottom(ctx, sym);
                return false;
            }
            else {
                sym->cls.version = version;
                return true;
            }
        case JIT_SYM_KNOWN_VALUE_TAG:
            if (Py_TYPE(sym->value.value)->tp_version_tag != version) {
                Py_CLEAR(sym->value.value);
                sym_set_bottom(ctx, sym);
                return false;
            };
            return true;
        case JIT_SYM_TUPLE_TAG:
            if (PyTuple_Type.tp_version_tag != version) {
                sym_set_bottom(ctx, sym);
                return false;
            };
            return true;
        case JIT_SYM_RECORDED_GEN_FUNC_TAG:
            if (PyGen_Type.tp_version_tag != version) {
                sym_set_bottom(ctx, sym);
                return false;
            }
            return true;
        case JIT_SYM_TYPE_VERSION_TAG:
            if (sym->version.version != version) {
                sym_set_bottom(ctx, sym);
                return false;
            }
            return true;
        case JIT_SYM_BOTTOM_TAG:
            return false;
        case JIT_SYM_NON_NULL_TAG:
        case JIT_SYM_UNKNOWN_TAG:
            sym->tag = JIT_SYM_TYPE_VERSION_TAG;
            sym->version.version = version;
            return true;
        case JIT_SYM_PREDICATE_TAG:
        case JIT_SYM_TRUTHINESS_TAG:
            if (version != PyBool_Type.tp_version_tag) {
                sym_set_bottom(ctx, sym);
                return false;
            }
            return true;
        case JIT_SYM_COMPACT_INT:
            if (version != PyLong_Type.tp_version_tag) {
                sym_set_bottom(ctx, sym);
                return false;
            }
            return true;
        case JIT_SYM_RECORDED_VALUE_TAG:
            if (Py_TYPE(sym->recorded_value.value)->tp_version_tag == version) {
                sym->recorded_value.known_type = true;
                sym->tag = JIT_SYM_KNOWN_CLASS_TAG;
                sym->cls.type = Py_TYPE(sym->recorded_value.value);
                sym->cls.version = version;
                return true;
            }
            else {
                sym_set_bottom(ctx, sym);
                return false;
            }
        case JIT_SYM_RECORDED_TYPE_TAG:
            if (sym->recorded_type.type->tp_version_tag == version) {
                sym->tag = JIT_SYM_KNOWN_CLASS_TAG;
                sym->cls.type = sym->recorded_type.type;
                sym->cls.version = version;
                return true;
            }
            else {
                sym_set_bottom(ctx, sym);
                return false;
            }
    }
    Py_UNREACHABLE();
}

void
_Py_uop_sym_set_const(JitOptContext *ctx, JitOptRef ref, PyObject *const_val)
{
    JitOptSymbol *sym = PyJitRef_Unwrap(ref);
    JitSymType tag = sym->tag;
    switch(tag) {
        case JIT_SYM_NULL_TAG:
        case JIT_SYM_RECORDED_GEN_FUNC_TAG:
            sym_set_bottom(ctx, sym);
            return;
        case JIT_SYM_KNOWN_CLASS_TAG:
            if (sym->cls.type != Py_TYPE(const_val)) {
                sym_set_bottom(ctx, sym);
                return;
            }
            make_const(sym, const_val);
            return;
        case JIT_SYM_KNOWN_VALUE_TAG:
            if (sym->value.value != const_val) {
                Py_CLEAR(sym->value.value);
                sym_set_bottom(ctx, sym);
            }
            return;
        case JIT_SYM_TUPLE_TAG:
            if (PyTuple_CheckExact(const_val)) {
                Py_ssize_t len = _Py_uop_sym_tuple_length(ref);
                if (len == PyTuple_GET_SIZE(const_val)) {
                    for (Py_ssize_t i = 0; i < len; i++) {
                        JitOptRef sym_item = _Py_uop_sym_tuple_getitem(ctx, ref, i);
                        PyObject *item = PyTuple_GET_ITEM(const_val, i);
                        _Py_uop_sym_set_const(ctx, sym_item, item);
                    }
                    make_const(sym, const_val);
                    return;
                }
            }
            sym_set_bottom(ctx, sym);
            return;
        case JIT_SYM_TYPE_VERSION_TAG:
            if (sym->version.version != Py_TYPE(const_val)->tp_version_tag) {
                sym_set_bottom(ctx, sym);
                return;
            }
            make_const(sym, const_val);
            return;
        case JIT_SYM_BOTTOM_TAG:
            return;
        case JIT_SYM_RECORDED_VALUE_TAG:
        case JIT_SYM_RECORDED_TYPE_TAG:
            /* The given value might contradict the recorded one,
             * in which case we could return bottom.
             * Just discard the recorded value for now */
        case JIT_SYM_NON_NULL_TAG:
        case JIT_SYM_UNKNOWN_TAG:
            make_const(sym, const_val);
            return;
        case JIT_SYM_PREDICATE_TAG:
            if (!PyBool_Check(const_val)) {
                sym_set_bottom(ctx, sym);
                return;
            }
            make_const(sym, const_val);
            return;
        case JIT_SYM_TRUTHINESS_TAG:
            if (!PyBool_Check(const_val) ||
                (_Py_uop_sym_is_const(ctx, ref) &&
                 _Py_uop_sym_get_const(ctx, ref) != const_val))
            {
                sym_set_bottom(ctx, sym);
                return;
            }
            JitOptRef value = PyJitRef_Wrap(
                allocation_base(ctx) + sym->truthiness.value);
            PyTypeObject *type = _Py_uop_sym_get_type(value);
            if (const_val == (sym->truthiness.invert ? Py_False : Py_True)) {
                // value is truthy. This is only useful for bool:
                if (type == &PyBool_Type) {
                    _Py_uop_sym_set_const(ctx, value, Py_True);
                }
            }
            // value is falsey:
            else if (type == &PyBool_Type) {
                _Py_uop_sym_set_const(ctx, value, Py_False);
            }
            else if (type == &PyLong_Type) {
                _Py_uop_sym_set_const(ctx, value, Py_GetConstant(Py_CONSTANT_ZERO));
            }
            else if (type == &PyUnicode_Type) {
                _Py_uop_sym_set_const(ctx, value, Py_GetConstant(Py_CONSTANT_EMPTY_STR));
            }
            // TODO: More types (GH-130415)!
            make_const(sym, const_val);
            return;
        case JIT_SYM_COMPACT_INT:
            if (_PyLong_CheckExactAndCompact(const_val)) {
                make_const(sym, const_val);
            }
            else {
                sym_set_bottom(ctx, sym);
            }
            return;
    }
}

void
_Py_uop_sym_set_null(JitOptContext *ctx, JitOptRef ref)
{
    JitOptSymbol *sym = PyJitRef_Unwrap(ref);
    if (sym->tag == JIT_SYM_UNKNOWN_TAG) {
        sym->tag = JIT_SYM_NULL_TAG;
    }
    else if (sym->tag > JIT_SYM_NULL_TAG) {
        sym_set_bottom(ctx, sym);
    }
}

void
_Py_uop_sym_set_non_null(JitOptContext *ctx, JitOptRef ref)
{
    JitOptSymbol *sym = PyJitRef_Unwrap(ref);
    if (sym->tag == JIT_SYM_UNKNOWN_TAG) {
        sym->tag = JIT_SYM_NON_NULL_TAG;
    }
    else if (sym->tag == JIT_SYM_NULL_TAG) {
        sym_set_bottom(ctx, sym);
    }
}

JitOptRef
_Py_uop_sym_new_unknown(JitOptContext *ctx)
{
    JitOptSymbol *res = sym_new(ctx);
    if (res == NULL) {
        return out_of_space_ref(ctx);
    }
    return PyJitRef_Wrap(res);
}

JitOptRef
_Py_uop_sym_new_not_null(JitOptContext *ctx)
{
    JitOptSymbol *res = sym_new(ctx);
    if (res == NULL) {
        return out_of_space_ref(ctx);
    }
    res->tag = JIT_SYM_NON_NULL_TAG;
    return PyJitRef_Wrap(res);
}

JitOptRef
_Py_uop_sym_new_type(JitOptContext *ctx, PyTypeObject *typ)
{
    JitOptSymbol *res = sym_new(ctx);
    if (res == NULL) {
        return out_of_space_ref(ctx);
    }
    JitOptRef ref = PyJitRef_Wrap(res);
    _Py_uop_sym_set_type(ctx, ref, typ);
    return ref;
}

// Adds a new reference to const_val, owned by the symbol.
JitOptRef
_Py_uop_sym_new_const(JitOptContext *ctx, PyObject *const_val)
{
    assert(const_val != NULL);
    JitOptSymbol *res = sym_new(ctx);
    if (res == NULL) {
        return out_of_space_ref(ctx);
    }
    JitOptRef ref = PyJitRef_Wrap(res);
    _Py_uop_sym_set_const(ctx, ref, const_val);
    return ref;
}

JitOptRef
_Py_uop_sym_new_const_steal(JitOptContext *ctx, PyObject *const_val)
{
    assert(const_val != NULL);
    JitOptRef res = _Py_uop_sym_new_const(ctx, const_val);
    // Decref once because sym_new_const increfs it.
    Py_DECREF(const_val);
    return res;
}

JitOptRef
_Py_uop_sym_new_null(JitOptContext *ctx)
{
    JitOptSymbol *null_sym = sym_new(ctx);
    if (null_sym == NULL) {
        return out_of_space_ref(ctx);
    }
    JitOptRef ref = PyJitRef_Wrap(null_sym);
    _Py_uop_sym_set_null(ctx, ref);
    return ref;
}

PyTypeObject *
_Py_uop_sym_get_type(JitOptRef ref)
{
    JitOptSymbol *sym = PyJitRef_Unwrap(ref);
    JitSymType tag = sym->tag;
    switch(tag) {
        case JIT_SYM_NULL_TAG:
        case JIT_SYM_BOTTOM_TAG:
        case JIT_SYM_NON_NULL_TAG:
        case JIT_SYM_UNKNOWN_TAG:
        case JIT_SYM_RECORDED_TYPE_TAG:
        case JIT_SYM_RECORDED_GEN_FUNC_TAG:
            return NULL;
        case JIT_SYM_RECORDED_VALUE_TAG:
            if (sym->recorded_value.known_type) {
                return Py_TYPE(sym->recorded_value.value);
            }
            return NULL;
        case JIT_SYM_KNOWN_CLASS_TAG:
            return sym->cls.type;
        case JIT_SYM_KNOWN_VALUE_TAG:
            return Py_TYPE(sym->value.value);
        case JIT_SYM_TYPE_VERSION_TAG:
            return _PyType_LookupByVersion(sym->version.version);
        case JIT_SYM_TUPLE_TAG:
            return &PyTuple_Type;
        case JIT_SYM_PREDICATE_TAG:
        case JIT_SYM_TRUTHINESS_TAG:
            return &PyBool_Type;
        case JIT_SYM_COMPACT_INT:
            return &PyLong_Type;
    }
    Py_UNREACHABLE();
}

PyTypeObject *
_Py_uop_sym_get_probable_type(JitOptRef ref)
{
    JitOptSymbol *sym = PyJitRef_Unwrap(ref);
    JitSymType tag = sym->tag;
    switch(tag) {
        case JIT_SYM_NULL_TAG:
        case JIT_SYM_BOTTOM_TAG:
        case JIT_SYM_NON_NULL_TAG:
        case JIT_SYM_UNKNOWN_TAG:
        case JIT_SYM_TYPE_VERSION_TAG:
        case JIT_SYM_TUPLE_TAG:
        case JIT_SYM_PREDICATE_TAG:
        case JIT_SYM_TRUTHINESS_TAG:
        case JIT_SYM_COMPACT_INT:
        case JIT_SYM_KNOWN_CLASS_TAG:
        case JIT_SYM_KNOWN_VALUE_TAG:
            return _Py_uop_sym_get_type(ref);
        case JIT_SYM_RECORDED_GEN_FUNC_TAG:
            return &PyGen_Type;
        case JIT_SYM_RECORDED_VALUE_TAG:
            return Py_TYPE(sym->recorded_value.value);
        case JIT_SYM_RECORDED_TYPE_TAG:
            return sym->recorded_type.type;
    }
    Py_UNREACHABLE();
}

unsigned int
_Py_uop_sym_get_type_version(JitOptRef ref)
{
    JitOptSymbol *sym = PyJitRef_Unwrap(ref);
    JitSymType tag = sym->tag;
    switch(tag) {
        case JIT_SYM_NULL_TAG:
        case JIT_SYM_BOTTOM_TAG:
        case JIT_SYM_NON_NULL_TAG:
        case JIT_SYM_UNKNOWN_TAG:
        case JIT_SYM_RECORDED_VALUE_TAG:
        case JIT_SYM_RECORDED_TYPE_TAG:
            return 0;
        case JIT_SYM_TYPE_VERSION_TAG:
            return sym->version.version;
        case JIT_SYM_KNOWN_CLASS_TAG:
            return sym->cls.version;
        case JIT_SYM_KNOWN_VALUE_TAG:
            return Py_TYPE(sym->value.value)->tp_version_tag;
        case JIT_SYM_TUPLE_TAG:
            return PyTuple_Type.tp_version_tag;
        case JIT_SYM_PREDICATE_TAG:
        case JIT_SYM_TRUTHINESS_TAG:
            return PyBool_Type.tp_version_tag;
        case JIT_SYM_COMPACT_INT:
            return PyLong_Type.tp_version_tag;
        case JIT_SYM_RECORDED_GEN_FUNC_TAG:
            return PyGen_Type.tp_version_tag;
    }
    Py_UNREACHABLE();
}


bool
_Py_uop_sym_has_type(JitOptRef sym)
{
    return _Py_uop_sym_get_type(sym) != NULL;
}

bool
_Py_uop_sym_matches_type(JitOptRef sym, PyTypeObject *typ)
{
    assert(typ != NULL && PyType_Check(typ));
    return _Py_uop_sym_get_type(sym) == typ;
}

bool
_Py_uop_sym_matches_type_version(JitOptRef sym, unsigned int version)
{
    return _Py_uop_sym_get_type_version(sym) == version;
}

PyObject *
_Py_uop_sym_get_probable_value(JitOptRef ref)
{
    JitOptSymbol *sym = PyJitRef_Unwrap(ref);
    JitSymType tag = sym->tag;
    switch(tag) {
        case JIT_SYM_NULL_TAG:
        case JIT_SYM_BOTTOM_TAG:
        case JIT_SYM_NON_NULL_TAG:
        case JIT_SYM_UNKNOWN_TAG:
        case JIT_SYM_RECORDED_TYPE_TAG:
        case JIT_SYM_TYPE_VERSION_TAG:
        case JIT_SYM_TUPLE_TAG:
        case JIT_SYM_PREDICATE_TAG:
        case JIT_SYM_TRUTHINESS_TAG:
        case JIT_SYM_COMPACT_INT:
        case JIT_SYM_KNOWN_CLASS_TAG:
        case JIT_SYM_RECORDED_GEN_FUNC_TAG:
            return NULL;
        case JIT_SYM_RECORDED_VALUE_TAG:
            return sym->recorded_value.value;
        case JIT_SYM_KNOWN_VALUE_TAG:
            return sym->value.value;
    }
    Py_UNREACHABLE();
}

PyCodeObject *
_Py_uop_sym_get_probable_func_code(JitOptRef ref)
{
    JitOptSymbol *sym = PyJitRef_Unwrap(ref);
    if (sym->tag == JIT_SYM_RECORDED_GEN_FUNC_TAG) {
        return (PyCodeObject *)PyFunction_GET_CODE(sym->recorded_gen_func.func);
    }
    PyObject *obj = _Py_uop_sym_get_probable_value(ref);
    if (obj != NULL) {
        if (PyFunction_Check(obj)) {
            return (PyCodeObject *)PyFunction_GET_CODE(obj);
        }
    }
    return NULL;
}

PyFunctionObject *
_Py_uop_sym_get_probable_function(JitOptRef ref)
{
    JitOptSymbol *sym = PyJitRef_Unwrap(ref);
    if (sym->tag == JIT_SYM_RECORDED_GEN_FUNC_TAG) {
        return sym->recorded_gen_func.func;
    }
    PyObject *obj = _Py_uop_sym_get_probable_value(ref);
    if (obj != NULL && PyFunction_Check(obj)) {
        return (PyFunctionObject *)obj;
    }
    return NULL;
}

int
_Py_uop_sym_truthiness(JitOptContext *ctx, JitOptRef ref)
{
    JitOptSymbol *sym = PyJitRef_Unwrap(ref);
    JitSymType tag = sym->tag;
    switch (tag) {
        case JIT_SYM_NULL_TAG:
        case JIT_SYM_TYPE_VERSION_TAG:
        case JIT_SYM_BOTTOM_TAG:
        case JIT_SYM_NON_NULL_TAG:
        case JIT_SYM_UNKNOWN_TAG:
        case JIT_SYM_COMPACT_INT:
        case JIT_SYM_PREDICATE_TAG:
        case JIT_SYM_RECORDED_VALUE_TAG:
        case JIT_SYM_RECORDED_TYPE_TAG:
        case JIT_SYM_RECORDED_GEN_FUNC_TAG:
            return -1;
        case JIT_SYM_KNOWN_CLASS_TAG:
            /* TODO :
             * Instances of some classes are always
             * true. We should return 1 in those cases */
            return -1;
        case JIT_SYM_KNOWN_VALUE_TAG:
            break;
        case JIT_SYM_TUPLE_TAG:
            return sym->tuple.length != 0;
        case JIT_SYM_TRUTHINESS_TAG:
        {
            JitOptSymbol *value = allocation_base(ctx) + sym->truthiness.value;
            int truthiness = _Py_uop_sym_truthiness(ctx,
                                                    PyJitRef_Wrap(value));
            if (truthiness < 0) {
                return truthiness;
            }
            truthiness ^= sym->truthiness.invert;
            make_const(sym, truthiness ? Py_True : Py_False);
            return truthiness;
        }
    }
    PyObject *value = sym->value.value;
    /* Only handle a few known safe types */
    if (value == Py_None) {
        return 0;
    }
    PyTypeObject *tp = Py_TYPE(value);
    if (tp == &PyLong_Type) {
        return !_PyLong_IsZero((PyLongObject *)value);
    }
    if (tp == &PyUnicode_Type) {
        return value != &_Py_STR(empty);
    }
    if (tp == &PyBool_Type) {
        return value == Py_True;
    }
    return -1;
}

JitOptRef
_Py_uop_sym_new_tuple(JitOptContext *ctx, int size, JitOptRef *args)
{
    JitOptSymbol *res = sym_new(ctx);
    if (res == NULL) {
        return out_of_space_ref(ctx);
    }
    if (size > MAX_SYMBOLIC_TUPLE_SIZE) {
        res->tag = JIT_SYM_KNOWN_CLASS_TAG;
        res->cls.type = &PyTuple_Type;
    }
    else {
        res->tag = JIT_SYM_TUPLE_TAG;
        res->tuple.length = size;
        for (int i = 0; i < size; i++) {
            res->tuple.items[i] = (uint16_t)(PyJitRef_Unwrap(args[i]) - allocation_base(ctx));
        }
    }
    return PyJitRef_Wrap(res);
}

JitOptRef
_Py_uop_sym_tuple_getitem(JitOptContext *ctx, JitOptRef ref, Py_ssize_t item)
{
    JitOptSymbol *sym = PyJitRef_Unwrap(ref);
    assert(item >= 0);
    if (sym->tag == JIT_SYM_KNOWN_VALUE_TAG) {
        PyObject *tuple = sym->value.value;
        if (PyTuple_CheckExact(tuple) && item < PyTuple_GET_SIZE(tuple)) {
            return _Py_uop_sym_new_const(ctx, PyTuple_GET_ITEM(tuple, item));
        }
    }
    else if (sym->tag == JIT_SYM_TUPLE_TAG && item < sym->tuple.length) {
        return PyJitRef_Wrap(allocation_base(ctx) + sym->tuple.items[item]);
    }
    return _Py_uop_sym_new_not_null(ctx);
}

Py_ssize_t
_Py_uop_sym_tuple_length(JitOptRef ref)
{
    JitOptSymbol *sym = PyJitRef_Unwrap(ref);
    if (sym->tag == JIT_SYM_KNOWN_VALUE_TAG) {
        PyObject *tuple = sym->value.value;
        if (PyTuple_CheckExact(tuple)) {
            return PyTuple_GET_SIZE(tuple);
        }
    }
    else if (sym->tag == JIT_SYM_TUPLE_TAG) {
        return sym->tuple.length;
    }
    return -1;
}

// Return true if known to be immortal.
bool
_Py_uop_symbol_is_immortal(JitOptSymbol *sym)
{
    if (sym->tag == JIT_SYM_KNOWN_VALUE_TAG) {
        return _Py_IsImmortal(sym->value.value);
    }
    if (sym->tag == JIT_SYM_KNOWN_CLASS_TAG) {
        return sym->cls.type == &PyBool_Type;
    }
    return false;
}

bool
_Py_uop_sym_is_compact_int(JitOptRef ref)
{
    JitOptSymbol *sym = PyJitRef_Unwrap(ref);
    if (sym->tag == JIT_SYM_KNOWN_VALUE_TAG) {
        return (bool)_PyLong_CheckExactAndCompact(sym->value.value);
    }
    return sym->tag == JIT_SYM_COMPACT_INT;
}

bool
_Py_uop_sym_is_immortal(JitOptRef ref)
{
    JitOptSymbol *sym = PyJitRef_Unwrap(ref);
    return _Py_uop_symbol_is_immortal(sym);
}

void
_Py_uop_sym_set_compact_int(JitOptContext *ctx, JitOptRef ref)
{
    JitOptSymbol *sym = PyJitRef_Unwrap(ref);
    JitSymType tag = sym->tag;
    switch(tag) {
        case JIT_SYM_NULL_TAG:
        case JIT_SYM_RECORDED_GEN_FUNC_TAG:
            sym_set_bottom(ctx, sym);
            return;
        case JIT_SYM_KNOWN_CLASS_TAG:
            if (sym->cls.type == &PyLong_Type) {
                sym->tag = JIT_SYM_COMPACT_INT;
            } else {
                sym_set_bottom(ctx, sym);
            }
            return;
        case JIT_SYM_TYPE_VERSION_TAG:
            if (sym->version.version == PyLong_Type.tp_version_tag) {
                sym->tag = JIT_SYM_COMPACT_INT;
            }
            else {
                sym_set_bottom(ctx, sym);
            }
            return;
        case JIT_SYM_KNOWN_VALUE_TAG:
            if (!_PyLong_CheckExactAndCompact(sym->value.value)) {
                Py_CLEAR(sym->value.value);
                sym_set_bottom(ctx, sym);
            }
            return;
        case JIT_SYM_TUPLE_TAG:
        case JIT_SYM_PREDICATE_TAG:
        case JIT_SYM_TRUTHINESS_TAG:
            sym_set_bottom(ctx, sym);
            return;
        case JIT_SYM_BOTTOM_TAG:
        case JIT_SYM_COMPACT_INT:
            return;
        case JIT_SYM_RECORDED_VALUE_TAG:
        case JIT_SYM_RECORDED_TYPE_TAG:
            /* The given value might contradict the recorded one,
             * in which case we could return bottom.
             * Just discard the recorded value for now */
        case JIT_SYM_NON_NULL_TAG:
        case JIT_SYM_UNKNOWN_TAG:
            sym->tag = JIT_SYM_COMPACT_INT;
            return;
    }
}

JitOptRef
_Py_uop_sym_new_predicate(JitOptContext *ctx, JitOptRef lhs_ref, JitOptRef rhs_ref, JitOptPredicateKind kind)
{
    JitOptSymbol *lhs = PyJitRef_Unwrap(lhs_ref);
    JitOptSymbol *rhs = PyJitRef_Unwrap(rhs_ref);

    JitOptSymbol *res = sym_new(ctx);
    if (res == NULL) {
        return out_of_space_ref(ctx);
    }

    res->tag = JIT_SYM_PREDICATE_TAG;
    res->predicate.kind = kind;
    res->predicate.lhs = (uint16_t)(lhs - allocation_base(ctx));
    res->predicate.rhs = (uint16_t)(rhs - allocation_base(ctx));

    return PyJitRef_Wrap(res);
}

void
_Py_uop_sym_apply_predicate_narrowing(JitOptContext *ctx, JitOptRef ref, bool branch_is_true)
{
    JitOptSymbol *sym = PyJitRef_Unwrap(ref);
    if (sym->tag != JIT_SYM_PREDICATE_TAG) {
        return;
    }

    JitOptPredicate pred = sym->predicate;

    JitOptRef lhs_ref = PyJitRef_Wrap(allocation_base(ctx) + pred.lhs);
    JitOptRef rhs_ref = PyJitRef_Wrap(allocation_base(ctx) + pred.rhs);

    bool lhs_is_const = _Py_uop_sym_is_const(ctx, lhs_ref);
    bool rhs_is_const = _Py_uop_sym_is_const(ctx, rhs_ref);
    if (!lhs_is_const && !rhs_is_const) {
        return;
    }

    bool narrow = false;
    switch(pred.kind) {
        case JIT_PRED_EQ:
        case JIT_PRED_IS:
            narrow = branch_is_true;
            break;
        case JIT_PRED_NE:
        case JIT_PRED_IS_NOT:
            narrow = !branch_is_true;
            break;
        default:
            return;
    }
    if (!narrow) {
        return;
    }

    JitOptRef subject_ref = lhs_is_const ? rhs_ref : lhs_ref;
    JitOptRef const_ref = lhs_is_const ? lhs_ref : rhs_ref;

    PyObject *const_val = _Py_uop_sym_get_const(ctx, const_ref);
    if (const_val == NULL) {
        return;
    }
    _Py_uop_sym_set_const(ctx, subject_ref, const_val);
    assert(_Py_uop_sym_is_const(ctx, subject_ref));
}

JitOptRef
_Py_uop_sym_new_truthiness(JitOptContext *ctx, JitOptRef ref, bool truthy)
{
    JitOptSymbol *value = PyJitRef_Unwrap(ref);
    // It's clearer to invert this in the signature:
    bool invert = !truthy;
    if (value->tag == JIT_SYM_TRUTHINESS_TAG && value->truthiness.invert == invert) {
        return ref;
    }
    JitOptSymbol *res = sym_new(ctx);
    if (res == NULL) {
        return out_of_space_ref(ctx);
    }
    int truthiness = _Py_uop_sym_truthiness(ctx, ref);
    if (truthiness < 0) {
        res->tag = JIT_SYM_TRUTHINESS_TAG;
        res->truthiness.invert = invert;
        res->truthiness.value = (uint16_t)(value - allocation_base(ctx));
    }
    else {
        make_const(res, (truthiness ^ invert) ? Py_True : Py_False);
    }
    return PyJitRef_Wrap(res);
}

JitOptRef
_Py_uop_sym_new_compact_int(JitOptContext *ctx)
{
    JitOptSymbol *sym = sym_new(ctx);
    if (sym == NULL) {
        return out_of_space_ref(ctx);
    }
    sym->tag = JIT_SYM_COMPACT_INT;
    return PyJitRef_Wrap(sym);
}

void
_Py_uop_sym_set_recorded_value(JitOptContext *ctx, JitOptRef ref, PyObject *value)
{
    // It is possible for value to be NULL due to respecialization
    // during execution of the traced instruction.
    if (value == NULL) {
        return;
    }
    JitOptSymbol *sym = PyJitRef_Unwrap(ref);
    JitSymType tag = sym->tag;
    switch(tag) {
        case JIT_SYM_NULL_TAG:
            sym_set_bottom(ctx, sym);
            return;
        case JIT_SYM_BOTTOM_TAG:
            return;
        case JIT_SYM_NON_NULL_TAG:
        case JIT_SYM_UNKNOWN_TAG:
            sym->tag = JIT_SYM_RECORDED_VALUE_TAG;
            sym->recorded_value.known_type = false;
            sym->recorded_value.value = value;
            return;
        case JIT_SYM_RECORDED_VALUE_TAG:
            if (sym->recorded_value.value != value) {
                sym_set_bottom(ctx, sym);
            }
            return;
        case JIT_SYM_RECORDED_TYPE_TAG:
            if (sym->recorded_type.type == Py_TYPE(value)) {
                sym->tag = JIT_SYM_RECORDED_VALUE_TAG;
                sym->recorded_value.known_type = false;
                sym->recorded_value.value = value;
            }
            else {
                sym_set_bottom(ctx, sym);
            }
            return;
        case JIT_SYM_KNOWN_CLASS_TAG:
            if (sym->cls.type == Py_TYPE(value)) {
                sym->tag = JIT_SYM_RECORDED_VALUE_TAG;
                sym->recorded_value.known_type = true;
                sym->recorded_value.value = value;
            }
            else {
                sym_set_bottom(ctx, sym);
            }
            return;
        case JIT_SYM_KNOWN_VALUE_TAG:
            return;
        case JIT_SYM_TYPE_VERSION_TAG:
            if (sym->version.version == Py_TYPE(value)->tp_version_tag) {
                sym->tag = JIT_SYM_RECORDED_VALUE_TAG;
                sym->recorded_value.known_type = true;
                sym->recorded_value.value = value;
            }
            else {
                sym_set_bottom(ctx, sym);
            }
            return;
        // In these cases the original information is more valuable
        case JIT_SYM_RECORDED_GEN_FUNC_TAG:
        case JIT_SYM_TUPLE_TAG:
        case JIT_SYM_PREDICATE_TAG:
        case JIT_SYM_TRUTHINESS_TAG:
        case JIT_SYM_COMPACT_INT:
            return;
    }
    Py_UNREACHABLE();
}
void
_Py_uop_sym_set_recorded_gen_func(JitOptContext *ctx, JitOptRef ref, PyFunctionObject *value)
{
    // It is possible for value to be NULL due to respecialization
    // during execution of the traced instruction.
    if (value == NULL) {
        return;
    }
    assert(!PyJitRef_IsNull(ref));
    JitOptSymbol *sym = PyJitRef_Unwrap(ref);
    JitSymType tag = sym->tag;
    switch(tag) {
        case JIT_SYM_NULL_TAG:
        case JIT_SYM_RECORDED_VALUE_TAG:
        case JIT_SYM_KNOWN_VALUE_TAG:
        case JIT_SYM_TUPLE_TAG:
        case JIT_SYM_PREDICATE_TAG:
        case JIT_SYM_TRUTHINESS_TAG:
        case JIT_SYM_COMPACT_INT:
        case JIT_SYM_BOTTOM_TAG:
            return;
        case JIT_SYM_NON_NULL_TAG:
        case JIT_SYM_UNKNOWN_TAG:
            sym->tag = JIT_SYM_RECORDED_GEN_FUNC_TAG;
            sym->recorded_gen_func.func = value;
            return;
        case JIT_SYM_RECORDED_TYPE_TAG:
            if (sym->recorded_type.type == &PyGen_Type) {
                sym->tag = JIT_SYM_RECORDED_GEN_FUNC_TAG;
                sym->recorded_gen_func.func = value;
            }
            else {
                sym_set_bottom(ctx, sym);
            }
            return;
        case JIT_SYM_KNOWN_CLASS_TAG:
            if (sym->cls.type == &PyGen_Type) {
                sym->tag = JIT_SYM_RECORDED_GEN_FUNC_TAG;
                sym->recorded_gen_func.func = value;
            }
            else {
                sym_set_bottom(ctx, sym);
            }
            return;
        case JIT_SYM_TYPE_VERSION_TAG:
            if (sym->version.version == PyGen_Type.tp_version_tag) {
                sym->tag = JIT_SYM_RECORDED_GEN_FUNC_TAG;
                sym->recorded_gen_func.func = value;
            }
            else {
                sym_set_bottom(ctx, sym);
            }
            return;
        case JIT_SYM_RECORDED_GEN_FUNC_TAG:
            if (sym->recorded_gen_func.func != value) {
                sym_set_bottom(ctx, sym);
            }
            return;
    }
    Py_UNREACHABLE();
}

void
_Py_uop_sym_set_recorded_type(JitOptContext *ctx, JitOptRef ref, PyTypeObject *type)
{
    // It is possible for type to be NULL due to respecialization
    // during execution of the traced instruction.
    if (type == NULL) {
        return;
    }
    assert(PyType_Check((PyObject *)type));
    JitOptSymbol *sym = PyJitRef_Unwrap(ref);
    JitSymType tag = sym->tag;
    switch(tag) {
        case JIT_SYM_NULL_TAG:
            sym_set_bottom(ctx, sym);
            return;
        case JIT_SYM_BOTTOM_TAG:
            return;
        case JIT_SYM_NON_NULL_TAG:
        case JIT_SYM_UNKNOWN_TAG:
            sym->tag = JIT_SYM_RECORDED_TYPE_TAG;
            sym->recorded_type.type = type;
            return;
        case JIT_SYM_RECORDED_VALUE_TAG:
            if (Py_TYPE(sym->recorded_value.value) != type) {
                sym_set_bottom(ctx, sym);
            }
            return;
        case JIT_SYM_RECORDED_TYPE_TAG:
            if (sym->recorded_type.type != type) {
                sym_set_bottom(ctx, sym);
            }
            return;
        case JIT_SYM_KNOWN_CLASS_TAG:
            return;
        case JIT_SYM_KNOWN_VALUE_TAG:
            return;
        case JIT_SYM_TYPE_VERSION_TAG:
            if (sym->version.version == type->tp_version_tag) {
                sym->tag = JIT_SYM_KNOWN_CLASS_TAG;
                sym->cls.type = type;
            }
            else {
                sym_set_bottom(ctx, sym);
            }
            return;
        // In these cases the original information is more valuable
        case JIT_SYM_TUPLE_TAG:
        case JIT_SYM_PREDICATE_TAG:
        case JIT_SYM_TRUTHINESS_TAG:
        case JIT_SYM_COMPACT_INT:
        case JIT_SYM_RECORDED_GEN_FUNC_TAG:
            return;
    }
    Py_UNREACHABLE();
}

// 0 on success, -1 on error.
_Py_UOpsAbstractFrame *
_Py_uop_frame_new_from_symbol(
    JitOptContext *ctx,
    JitOptRef callable,
    JitOptRef *args,
    int arg_len)
{
    PyCodeObject *co = _Py_uop_sym_get_probable_func_code(callable);
    if (co == NULL) {
        ctx->done = true;
        return NULL;
    }
    _Py_UOpsAbstractFrame *frame = _Py_uop_frame_new(ctx, co, args, arg_len);
    if (frame == NULL) {
        return NULL;
    }
    PyFunctionObject *func = _Py_uop_sym_get_probable_function(callable);
    if (func != NULL) {
        assert(PyFunction_Check(func));
        frame->func = func;
    }
    assert(frame->stack_pointer != NULL);
    frame->callable = callable;
    return frame;
}

// 0 on success, -1 on error.
_Py_UOpsAbstractFrame *
_Py_uop_frame_new(
    JitOptContext *ctx,
    PyCodeObject *co,
    JitOptRef *args,
    int arg_len)
{
    assert(co != NULL);
    if (ctx->curr_frame_depth >= MAX_ABSTRACT_FRAME_DEPTH) {
        ctx->done = true;
        ctx->out_of_space = true;
        OPT_STAT_INC(optimizer_frame_overflow);
        return NULL;
    }
    _Py_UOpsAbstractFrame *frame = &ctx->frames[ctx->curr_frame_depth];
    frame->code = co;

    frame->locals = ctx->locals.used;
    ctx->locals.used += co->co_nlocalsplus;
    frame->locals_len = co->co_nlocalsplus;

    frame->stack = ctx->stack.used;
    ctx->stack.used += co->co_stacksize;
    frame->stack_len = co->co_stacksize;

    frame->stack_pointer = frame->stack;
    frame->globals_checked_version = 0;
    frame->globals_watched = false;
    frame->builtins_checked = false;
    frame->func = NULL;
    frame->caller = false;
    frame->is_c_recursion_checked = false;
    if (ctx->locals.used > ctx->locals.end || ctx->stack.used > ctx->stack.end) {
        ctx->done = true;
        ctx->out_of_space = true;
        return NULL;
    }

    // Initialize with the initial state of all local variables
    arg_len = Py_MIN(arg_len, co->co_argcount);
    for (int i = 0; i < arg_len; i++) {
        frame->locals[i] = PyJitRef_RemoveUnique(args[i]);
    }

    // When args is available, missing parameters get defaults or *args/**kwargs.
    // Other locals start as NULL. When args is NULL, treat all locals as unknown.
    int parameter_count = co->co_argcount + co->co_kwonlyargcount;
    parameter_count += (co->co_flags & CO_VARARGS) != 0;
    parameter_count += (co->co_flags & CO_VARKEYWORDS) != 0;

    for (int i = arg_len; i < co->co_nlocalsplus; i++) {
        if (args == NULL) {
            frame->locals[i] = _Py_uop_sym_new_unknown(ctx);
        }
        else if (i < parameter_count) {
            frame->locals[i] = _Py_uop_sym_new_not_null(ctx);
        }
        else {
            frame->locals[i] = _Py_uop_sym_new_null(ctx);
        }
    }

    frame->callable = _Py_uop_sym_new_not_null(ctx);

    /* Most optimizations rely on code objects being immutable (including sys._getframe modifications),
     * and up to date for instrumentation. */
    _Py_BloomFilter_Add(ctx->dependencies, co);

    assert(frame->locals != NULL);
    return frame;
}

JitOptRef *
_Py_uop_sym_set_stack_depth(JitOptContext *ctx, int stack_depth, JitOptRef *current_sp) {
    _Py_UOpsAbstractFrame *frame = ctx->frame;
    assert(frame->stack != NULL);
    JitOptRef *new_stack_pointer = frame->stack + stack_depth;
    if (current_sp > new_stack_pointer) {
        ctx->done = true;
        ctx->contradiction = true;
        return NULL;
    }
    if (new_stack_pointer > ctx->stack.end) {
        ctx->done = true;
        ctx->out_of_space = true;
        return NULL;
    }
    int delta = (int)(new_stack_pointer - current_sp);
    assert(delta >= 0);
    if (delta) {
        /* Shift existing stack elements up */
        for (JitOptRef *p = current_sp-1; p >= frame->stack; p--) {
            p[delta] = *p;
        }
        /* Fill rest of stack with unknowns */
        for (int i = 0; i < delta; i++) {
            frame->stack[i] = _Py_uop_sym_new_unknown(ctx);
        }
    }
    return frame->stack_pointer = new_stack_pointer;
}


void
_Py_uop_abstractcontext_fini(JitOptContext *ctx)
{
    if (ctx == NULL) {
        return;
    }
    ctx->curr_frame_depth = 0;
    int tys = ctx->t_arena.ty_curr_number;
    for (int i = 0; i < tys; i++) {
        JitOptSymbol *sym = &ctx->t_arena.arena[i];
        if (sym->tag == JIT_SYM_KNOWN_VALUE_TAG) {
            Py_CLEAR(sym->value.value);
        }
    }
}

// Leave a bit of space to push values before checking that there is space for a new frame
#define STACK_HEADROOM 2

void
_Py_uop_abstractcontext_init(JitOptContext *ctx, _PyBloomFilter *dependencies)
{
    static_assert(sizeof(JitOptSymbol) <= 3 * sizeof(uint64_t), "JitOptSymbol has grown");
    static_assert(VIRTUAL_ARENA_SIZE <= UINT16_MAX,
                  "virtual-object arena does not fit in JitVirtualRef");

    ctx->stack.used = ctx->stack_array;
    ctx->stack.end = &ctx->stack_array[ABSTRACT_INTERP_STACK_SIZE-STACK_HEADROOM];
    ctx->locals.used = ctx->locals_array;
    ctx->locals.end = &ctx->locals_array[ABSTRACT_INTERP_LOCALS_SIZE-STACK_HEADROOM];
#ifdef Py_DEBUG // Aids debugging a little. There should never be NULL in the abstract interpreter.
    for (int i = 0 ; i < ABSTRACT_INTERP_STACK_SIZE; i++) {
        ctx->stack_array[i] = PyJitRef_NULL;
    }
    for (int i = 0 ; i < ABSTRACT_INTERP_LOCALS_SIZE; i++) {
        ctx->locals_array[i] = PyJitRef_NULL;
    }
#endif

    // Setup the arena for sym expressions.
    ctx->t_arena.ty_curr_number = 0;
    ctx->t_arena.ty_max_number = TY_ARENA_SIZE;
    ctx->virtuals.virtual_curr_number = 0;
    ctx->virtuals.virtual_max_number = VIRTUAL_ARENA_SIZE;
    ctx->virtual_floats_enabled = false;
    ctx->retry_without_virtuals = false;

    // Frame setup
    ctx->curr_frame_depth = 0;

    // Ctx signals.
    // Note: this must happen before frame_new, as it might override
    // the result should frame_new set things to bottom.
    ctx->done = false;
    ctx->out_of_space = false;
    ctx->contradiction = false;
    ctx->builtins_watched = false;
    ctx->dependencies = dependencies;
}

int
_Py_uop_frame_pop(JitOptContext *ctx, PyCodeObject *co)
{
    _Py_UOpsAbstractFrame *frame = ctx->frame;
    ctx->stack.used = frame->stack;
    ctx->locals.used = frame->locals;

    ctx->curr_frame_depth--;

    if (ctx->curr_frame_depth >= 1) {
        ctx->frame = &ctx->frames[ctx->curr_frame_depth - 1];
        assert(ctx->frame->locals != NULL);

        // We returned to the correct code. Nothing to do here.
        if (co == ctx->frame->code) {
            return 0;
        }
        // Else: the code we recorded doesn't match the code we *think* we're
        // returning to. We could trace anything, we can't just return to the
        // old frame. We have to restore what the tracer recorded
        // as the traced next frame.
        // Remove the current frame, and later swap it out with the right one.
        else {
            ctx->curr_frame_depth--;
        }
    }
    // Else: trace stack underflow.

    // This handles swapping out frames.
    _Py_UOpsAbstractFrame *new_frame = _Py_uop_frame_new(ctx, co, NULL, 0);
    if (new_frame == NULL) {
        ctx->done = true;
        return 1;
    }

    ctx->curr_frame_depth++;
    ctx->frame = new_frame;
    assert(ctx->frame->locals != NULL);

    return 0;
}

#define TEST_PREDICATE(PRED, MSG) \
do { \
    if (!(PRED)) { \
        PyErr_SetString( \
            PyExc_AssertionError, \
            (MSG)); \
        goto fail; \
    } \
} while (0)

static JitOptSymbol *
make_bottom(JitOptContext *ctx)
{
    JitOptSymbol *sym = sym_new(ctx);
    if (sym == NULL) {
        return out_of_space(ctx);
    }
    sym->tag = JIT_SYM_BOTTOM_TAG;
    return sym;
}

PyObject *
_Py_uop_symbols_test(PyObject *Py_UNUSED(self), PyObject *Py_UNUSED(ignored))
{
    JitOptContext context;
    JitOptContext *ctx = &context;
    _Py_uop_abstractcontext_init(ctx, NULL);
    PyObject *val_42 = NULL;
    PyObject *val_43 = NULL;
    PyObject *val_big = NULL;
    PyObject *tuple = NULL;
    PyFunctionObject *func = NULL;

    // Use a single 'sym' variable so copy-pasting tests is easier.
    JitOptRef ref = _Py_uop_sym_new_unknown(ctx);
    if (PyJitRef_IsNull(ref)) {
        goto fail;
    }
    TEST_PREDICATE(!_Py_uop_sym_is_null(ref), "top is NULL");
    TEST_PREDICATE(!_Py_uop_sym_is_not_null(ref), "top is not NULL");
    TEST_PREDICATE(!_Py_uop_sym_matches_type(ref, &PyLong_Type), "top matches a type");
    TEST_PREDICATE(!_Py_uop_sym_is_const(ctx, ref), "top is a constant");
    TEST_PREDICATE(_Py_uop_sym_get_const(ctx, ref) == NULL, "top as constant is not NULL");
    TEST_PREDICATE(!_Py_uop_sym_is_bottom(ref), "top is bottom");

    ref = PyJitRef_Wrap(make_bottom(ctx));
    if (PyJitRef_IsNull(ref)) {
        goto fail;
    }
    TEST_PREDICATE(!_Py_uop_sym_is_null(ref), "bottom is NULL is not false");
    TEST_PREDICATE(!_Py_uop_sym_is_not_null(ref), "bottom is not NULL is not false");
    TEST_PREDICATE(!_Py_uop_sym_matches_type(ref, &PyLong_Type), "bottom matches a type");
    TEST_PREDICATE(!_Py_uop_sym_is_const(ctx, ref), "bottom is a constant is not false");
    TEST_PREDICATE(_Py_uop_sym_get_const(ctx, ref) == NULL, "bottom as constant is not NULL");
    TEST_PREDICATE(_Py_uop_sym_is_bottom(ref), "bottom isn't bottom");

    ref = _Py_uop_sym_new_type(ctx, &PyLong_Type);
    if (PyJitRef_IsNull(ref)) {
        goto fail;
    }
    TEST_PREDICATE(!_Py_uop_sym_is_null(ref), "int is NULL");
    TEST_PREDICATE(_Py_uop_sym_is_not_null(ref), "int isn't not NULL");
    TEST_PREDICATE(_Py_uop_sym_matches_type(ref, &PyLong_Type), "int isn't int");
    TEST_PREDICATE(!_Py_uop_sym_matches_type(ref, &PyFloat_Type), "int matches float");
    TEST_PREDICATE(!_Py_uop_sym_is_const(ctx, ref), "int is a constant");
    TEST_PREDICATE(_Py_uop_sym_get_const(ctx, ref) == NULL, "int as constant is not NULL");

    _Py_uop_sym_set_type(ctx, ref, &PyLong_Type);  // Should be a no-op
    TEST_PREDICATE(_Py_uop_sym_matches_type(ref, &PyLong_Type), "(int and int) isn't int");

    _Py_uop_sym_set_type(ctx, ref, &PyFloat_Type);  // Should make it bottom
    TEST_PREDICATE(_Py_uop_sym_is_bottom(ref), "(int and float) isn't bottom");

    val_42 = PyLong_FromLong(42);
    assert(val_42 != NULL);
    assert(_Py_IsImmortal(val_42));

    val_43 = PyLong_FromLong(43);
    assert(val_43 != NULL);
    assert(_Py_IsImmortal(val_43));

    ref = _Py_uop_sym_new_type(ctx, &PyLong_Type);
    if (PyJitRef_IsNull(ref)) {
        goto fail;
    }
    _Py_uop_sym_set_const(ctx, ref, val_42);
    TEST_PREDICATE(_Py_uop_sym_truthiness(ctx, ref) == 1, "bool(42) is not True");
    TEST_PREDICATE(!_Py_uop_sym_is_null(ref), "42 is NULL");
    TEST_PREDICATE(_Py_uop_sym_is_not_null(ref), "42 isn't not NULL");
    TEST_PREDICATE(_Py_uop_sym_matches_type(ref, &PyLong_Type), "42 isn't an int");
    TEST_PREDICATE(!_Py_uop_sym_matches_type(ref, &PyFloat_Type), "42 matches float");
    TEST_PREDICATE(_Py_uop_sym_is_const(ctx, ref), "42 is not a constant");
    TEST_PREDICATE(_Py_uop_sym_get_const(ctx, ref) != NULL, "42 as constant is NULL");
    TEST_PREDICATE(_Py_uop_sym_get_const(ctx, ref) == val_42, "42 as constant isn't 42");
    TEST_PREDICATE(_Py_uop_sym_is_immortal(ref), "42 is not immortal");

    _Py_uop_sym_set_type(ctx, ref, &PyLong_Type);  // Should be a no-op
    TEST_PREDICATE(_Py_uop_sym_matches_type(ref, &PyLong_Type), "(42 and 42) isn't an int");
    TEST_PREDICATE(_Py_uop_sym_get_const(ctx, ref) == val_42, "(42 and 42) as constant isn't 42");

    _Py_uop_sym_set_type(ctx, ref, &PyFloat_Type);  // Should make it bottom
    TEST_PREDICATE(_Py_uop_sym_is_bottom(ref), "(42 and float) isn't bottom");

    ref = _Py_uop_sym_new_type(ctx, &PyBool_Type);
    TEST_PREDICATE(_Py_uop_sym_is_immortal(ref), "a bool is not immortal");

    ref = _Py_uop_sym_new_type(ctx, &PyLong_Type);
    if (PyJitRef_IsNull(ref)) {
        goto fail;
    }
    _Py_uop_sym_set_const(ctx, ref, val_42);
    _Py_uop_sym_set_const(ctx, ref, val_43);  // Should make it bottom
    TEST_PREDICATE(_Py_uop_sym_is_bottom(ref), "(42 and 43) isn't bottom");


    ref = _Py_uop_sym_new_const(ctx, Py_None);
    TEST_PREDICATE(_Py_uop_sym_truthiness(ctx, ref) == 0, "bool(None) is not False");
    ref = _Py_uop_sym_new_const(ctx, Py_False);
    TEST_PREDICATE(_Py_uop_sym_truthiness(ctx, ref) == 0, "bool(False) is not False");
    ref = _Py_uop_sym_new_const(ctx, PyLong_FromLong(0));
    TEST_PREDICATE(_Py_uop_sym_truthiness(ctx, ref) == 0, "bool(0) is not False");

    JitOptRef i1 = _Py_uop_sym_new_type(ctx, &PyFloat_Type);
    JitOptRef i2 = _Py_uop_sym_new_const(ctx, val_43);
    JitOptRef array[2] = { i1, i2 };
    ref = _Py_uop_sym_new_tuple(ctx, 2, array);
    TEST_PREDICATE(
        _Py_uop_sym_matches_type(_Py_uop_sym_tuple_getitem(ctx, ref, 0), &PyFloat_Type),
        "tuple item does not match value used to create tuple"
    );
    TEST_PREDICATE(
        _Py_uop_sym_get_const(ctx, _Py_uop_sym_tuple_getitem(ctx, ref, 1)) == val_43,
        "tuple item does not match value used to create tuple"
    );
    PyObject *pair[2] = { val_42, val_43 };
    tuple = PyTuple_FromArray(pair, 2);
    ref = _Py_uop_sym_new_const(ctx, tuple);
    TEST_PREDICATE(
        _Py_uop_sym_get_const(ctx, _Py_uop_sym_tuple_getitem(ctx, ref, 1)) == val_43,
        "tuple item does not match value used to create tuple"
    );
    ref = _Py_uop_sym_new_type(ctx, &PyTuple_Type);
    TEST_PREDICATE(
        _Py_uop_sym_is_not_null(_Py_uop_sym_tuple_getitem(ctx, ref, 42)),
        "Unknown tuple item is not narrowed to non-NULL"
    );
    JitOptRef value = _Py_uop_sym_new_type(ctx, &PyBool_Type);
    ref = _Py_uop_sym_new_truthiness(ctx, value, false);
    TEST_PREDICATE(_Py_uop_sym_matches_type(ref, &PyBool_Type), "truthiness is not boolean");
    TEST_PREDICATE(_Py_uop_sym_truthiness(ctx, ref) == -1, "truthiness is not unknown");
    TEST_PREDICATE(_Py_uop_sym_is_const(ctx, ref) == false, "truthiness is constant");
    TEST_PREDICATE(_Py_uop_sym_get_const(ctx, ref) == NULL, "truthiness is not NULL");
    TEST_PREDICATE(_Py_uop_sym_is_const(ctx, value) == false, "value is constant");
    TEST_PREDICATE(_Py_uop_sym_get_const(ctx, value) == NULL, "value is not NULL");
    _Py_uop_sym_set_const(ctx, ref, Py_False);
    TEST_PREDICATE(_Py_uop_sym_matches_type(ref, &PyBool_Type), "truthiness is not boolean");
    TEST_PREDICATE(_Py_uop_sym_truthiness(ctx, ref) == 0, "truthiness is not True");
    TEST_PREDICATE(_Py_uop_sym_is_const(ctx, ref) == true, "truthiness is not constant");
    TEST_PREDICATE(_Py_uop_sym_get_const(ctx, ref) == Py_False, "truthiness is not False");
    TEST_PREDICATE(_Py_uop_sym_is_const(ctx, value) == true, "value is not constant");
    TEST_PREDICATE(_Py_uop_sym_get_const(ctx, value) == Py_True, "value is not True");

    // Resolving predicate result to True should narrow subject to True
    JitOptRef subject = _Py_uop_sym_new_unknown(ctx);
    JitOptRef const_true = _Py_uop_sym_new_const(ctx, Py_True);
    if (PyJitRef_IsNull(subject) || PyJitRef_IsNull(const_true)) {
        goto fail;
    }
    ref = _Py_uop_sym_new_predicate(ctx, subject, const_true, JIT_PRED_IS);
    if (PyJitRef_IsNull(ref)) {
        goto fail;
    }
    _Py_uop_sym_apply_predicate_narrowing(ctx, ref, true);
    TEST_PREDICATE(_Py_uop_sym_is_const(ctx, subject), "predicate narrowing did not const-narrow subject");
    TEST_PREDICATE(_Py_uop_sym_get_const(ctx, subject) == Py_True, "predicate narrowing did not narrow subject to True");

    // Resolving predicate result to False should not narrow subject
    subject = _Py_uop_sym_new_unknown(ctx);
    if (PyJitRef_IsNull(subject)) {
        goto fail;
    }
    ref = _Py_uop_sym_new_predicate(ctx, subject, const_true, JIT_PRED_IS);
    if (PyJitRef_IsNull(ref)) {
        goto fail;
    }
    _Py_uop_sym_apply_predicate_narrowing(ctx, ref, false);
    TEST_PREDICATE(!_Py_uop_sym_is_const(ctx, subject), "predicate narrowing incorrectly narrowed subject");

    // Resolving inverted predicate to False should narrow subject to True
    subject = _Py_uop_sym_new_unknown(ctx);
    if (PyJitRef_IsNull(subject)) {
        goto fail;
    }
    ref = _Py_uop_sym_new_predicate(ctx, subject, const_true, JIT_PRED_IS_NOT);
    if (PyJitRef_IsNull(ref)) {
        goto fail;
    }
    _Py_uop_sym_apply_predicate_narrowing(ctx, ref, false);
    TEST_PREDICATE(_Py_uop_sym_is_const(ctx, subject), "predicate narrowing (inverted) did not const-narrow subject");
    TEST_PREDICATE(_Py_uop_sym_get_const(ctx, subject) == Py_True, "predicate narrowing (inverted) did not narrow subject to True");

    // Resolving inverted predicate to True should not narrow subject
    subject = _Py_uop_sym_new_unknown(ctx);
    if (PyJitRef_IsNull(subject)) {
        goto fail;
    }
    ref = _Py_uop_sym_new_predicate(ctx, subject, const_true, JIT_PRED_IS_NOT);
    if (PyJitRef_IsNull(ref)) {
        goto fail;
    }
    _Py_uop_sym_apply_predicate_narrowing(ctx, ref, true);
    TEST_PREDICATE(!_Py_uop_sym_is_const(ctx, subject), "predicate narrowing incorrectly narrowed subject (inverted/true)");

    // Test narrowing subject to None
    subject = _Py_uop_sym_new_unknown(ctx);
    JitOptRef const_none = _Py_uop_sym_new_const(ctx, Py_None);
    if (PyJitRef_IsNull(subject) || PyJitRef_IsNull(const_none)) {
        goto fail;
    }
    ref = _Py_uop_sym_new_predicate(ctx, subject, const_none, JIT_PRED_IS);
    if (PyJitRef_IsNull(ref)) {
        goto fail;
    }
    _Py_uop_sym_apply_predicate_narrowing(ctx, ref, true);
    TEST_PREDICATE(_Py_uop_sym_is_const(ctx, subject), "predicate narrowing did not const-narrow subject (None)");
    TEST_PREDICATE(_Py_uop_sym_get_const(ctx, subject) == Py_None, "predicate narrowing did not narrow subject to None");

    // Test narrowing subject to numerical constant from is comparison
    subject = _Py_uop_sym_new_unknown(ctx);
    PyObject *one_obj = PyLong_FromLong(1);
    JitOptRef const_one = _Py_uop_sym_new_const(ctx, one_obj);
    if (PyJitRef_IsNull(subject) || one_obj == NULL || PyJitRef_IsNull(const_one)) {
        goto fail;
    }
    ref = _Py_uop_sym_new_predicate(ctx, subject, const_one, JIT_PRED_IS);
    if (PyJitRef_IsNull(ref)) {
        goto fail;
    }
    _Py_uop_sym_apply_predicate_narrowing(ctx, ref, true);
    TEST_PREDICATE(_Py_uop_sym_is_const(ctx, subject), "predicate narrowing did not const-narrow subject (1)");
    TEST_PREDICATE(_Py_uop_sym_get_const(ctx, subject) == one_obj, "predicate narrowing did not narrow subject to 1");

    // Test narrowing subject to constant from EQ predicate for int
    subject = _Py_uop_sym_new_unknown(ctx);
    if (PyJitRef_IsNull(subject)) {
        goto fail;
    }
    ref = _Py_uop_sym_new_predicate(ctx, subject, const_one, JIT_PRED_EQ);
    if (PyJitRef_IsNull(ref)) {
        goto fail;
    }
    _Py_uop_sym_apply_predicate_narrowing(ctx, ref, true);
    TEST_PREDICATE(_Py_uop_sym_is_const(ctx, subject), "predicate narrowing did not const-narrow subject (1)");
    TEST_PREDICATE(_Py_uop_sym_get_const(ctx, subject) == one_obj, "predicate narrowing did not narrow subject to 1");

    // Resolving EQ predicate to False should not narrow subject for int
    subject = _Py_uop_sym_new_unknown(ctx);
    if (PyJitRef_IsNull(subject)) {
        goto fail;
    }
    ref = _Py_uop_sym_new_predicate(ctx, subject, const_one, JIT_PRED_EQ);
    if (PyJitRef_IsNull(ref)) {
        goto fail;
    }
    _Py_uop_sym_apply_predicate_narrowing(ctx, ref, false);
    TEST_PREDICATE(!_Py_uop_sym_is_const(ctx, subject), "predicate narrowing incorrectly narrowed subject (inverted/true)");

    // Test narrowing subject to constant from NE predicate for int
    subject = _Py_uop_sym_new_unknown(ctx);
    if (PyJitRef_IsNull(subject)) {
        goto fail;
    }
    ref = _Py_uop_sym_new_predicate(ctx, subject, const_one, JIT_PRED_NE);
    if (PyJitRef_IsNull(ref)) {
        goto fail;
    }
    _Py_uop_sym_apply_predicate_narrowing(ctx, ref, false);
    TEST_PREDICATE(_Py_uop_sym_is_const(ctx, subject), "predicate narrowing did not const-narrow subject (1)");
    TEST_PREDICATE(_Py_uop_sym_get_const(ctx, subject) == one_obj, "predicate narrowing did not narrow subject to 1");

    // Resolving NE predicate to true should not narrow subject for int
    subject = _Py_uop_sym_new_unknown(ctx);
    if (PyJitRef_IsNull(subject)) {
        goto fail;
    }
    ref = _Py_uop_sym_new_predicate(ctx, subject, const_one, JIT_PRED_NE);
    if (PyJitRef_IsNull(ref)) {
        goto fail;
    }
    _Py_uop_sym_apply_predicate_narrowing(ctx, ref, true);
    TEST_PREDICATE(!_Py_uop_sym_is_const(ctx, subject), "predicate narrowing incorrectly narrowed subject (inverted/true)");

    // Test narrowing subject to constant from EQ predicate for float
    subject = _Py_uop_sym_new_unknown(ctx);
    PyObject *float_tenth_obj = PyFloat_FromDouble(0.1);
    JitOptRef const_float_tenth = _Py_uop_sym_new_const(ctx, float_tenth_obj);
    if (PyJitRef_IsNull(subject) || float_tenth_obj == NULL || PyJitRef_IsNull(const_float_tenth)) {
        goto fail;
    }
    ref = _Py_uop_sym_new_predicate(ctx, subject, const_float_tenth, JIT_PRED_EQ);
    if (PyJitRef_IsNull(ref)) {
        goto fail;
    }
    _Py_uop_sym_apply_predicate_narrowing(ctx, ref, true);
    TEST_PREDICATE(_Py_uop_sym_is_const(ctx, subject), "predicate narrowing did not const-narrow subject (float)");
    TEST_PREDICATE(_Py_uop_sym_get_const(ctx, subject) == float_tenth_obj, "predicate narrowing did not narrow subject to 0.1");

    // Resolving EQ predicate to False should not narrow subject for float
    subject = _Py_uop_sym_new_unknown(ctx);
    if (PyJitRef_IsNull(subject)) {
        goto fail;
    }
    ref = _Py_uop_sym_new_predicate(ctx, subject, const_float_tenth, JIT_PRED_EQ);
    if (PyJitRef_IsNull(ref)) {
        goto fail;
    }
    _Py_uop_sym_apply_predicate_narrowing(ctx, ref, false);
    TEST_PREDICATE(!_Py_uop_sym_is_const(ctx, subject), "predicate narrowing incorrectly narrowed subject (inverted/true)");

    // Test narrowing subject to constant from NE predicate for float
    subject = _Py_uop_sym_new_unknown(ctx);
    if (PyJitRef_IsNull(subject)) {
        goto fail;
    }
    ref = _Py_uop_sym_new_predicate(ctx, subject, const_float_tenth, JIT_PRED_NE);
    if (PyJitRef_IsNull(ref)) {
        goto fail;
    }
    _Py_uop_sym_apply_predicate_narrowing(ctx, ref, false);
    TEST_PREDICATE(_Py_uop_sym_is_const(ctx, subject), "predicate narrowing did not const-narrow subject (float)");
    TEST_PREDICATE(_Py_uop_sym_get_const(ctx, subject) == float_tenth_obj, "predicate narrowing did not narrow subject to 0.1");

    // Resolving NE predicate to true should not narrow subject for float
    subject = _Py_uop_sym_new_unknown(ctx);
    if (PyJitRef_IsNull(subject)) {
        goto fail;
    }
    ref = _Py_uop_sym_new_predicate(ctx, subject, const_float_tenth, JIT_PRED_NE);
    if (PyJitRef_IsNull(ref)) {
        goto fail;
    }
    _Py_uop_sym_apply_predicate_narrowing(ctx, ref, true);
    TEST_PREDICATE(!_Py_uop_sym_is_const(ctx, subject), "predicate narrowing incorrectly narrowed subject (inverted/true)");

    // Test narrowing subject to constant from EQ predicate for str
    subject = _Py_uop_sym_new_unknown(ctx);
    PyObject *str_hello_obj = PyUnicode_FromString("hello");
    JitOptRef const_str_hello = _Py_uop_sym_new_const(ctx, str_hello_obj);
    if (PyJitRef_IsNull(subject) || str_hello_obj == NULL || PyJitRef_IsNull(const_str_hello)) {
        goto fail;
    }
    ref = _Py_uop_sym_new_predicate(ctx, subject, const_str_hello, JIT_PRED_EQ);
    if (PyJitRef_IsNull(ref)) {
        goto fail;
    }
    _Py_uop_sym_apply_predicate_narrowing(ctx, ref, true);
    TEST_PREDICATE(_Py_uop_sym_is_const(ctx, subject), "predicate narrowing did not const-narrow subject (str)");
    TEST_PREDICATE(_Py_uop_sym_get_const(ctx, subject) == str_hello_obj, "predicate narrowing did not narrow subject to hello");

    // Resolving EQ predicate to False should not narrow subject for str
    subject = _Py_uop_sym_new_unknown(ctx);
    if (PyJitRef_IsNull(subject)) {
        goto fail;
    }
    ref = _Py_uop_sym_new_predicate(ctx, subject, const_str_hello, JIT_PRED_EQ);
    if (PyJitRef_IsNull(ref)) {
        goto fail;
    }
    _Py_uop_sym_apply_predicate_narrowing(ctx, ref, false);
    TEST_PREDICATE(!_Py_uop_sym_is_const(ctx, subject), "predicate narrowing incorrectly narrowed subject (inverted/true)");

    // Test narrowing subject to constant from NE predicate for str
    subject = _Py_uop_sym_new_unknown(ctx);
    if (PyJitRef_IsNull(subject)) {
        goto fail;
    }
    ref = _Py_uop_sym_new_predicate(ctx, subject, const_str_hello, JIT_PRED_NE);
    if (PyJitRef_IsNull(ref)) {
        goto fail;
    }
    _Py_uop_sym_apply_predicate_narrowing(ctx, ref, false);
    TEST_PREDICATE(_Py_uop_sym_is_const(ctx, subject), "predicate narrowing did not const-narrow subject (str)");
    TEST_PREDICATE(_Py_uop_sym_get_const(ctx, subject) == str_hello_obj, "predicate narrowing did not narrow subject to hello");

    // Resolving NE predicate to true should not narrow subject for str
    subject = _Py_uop_sym_new_unknown(ctx);
    if (PyJitRef_IsNull(subject)) {
        goto fail;
    }
    ref = _Py_uop_sym_new_predicate(ctx, subject, const_str_hello, JIT_PRED_NE);
    if (PyJitRef_IsNull(ref)) {
        goto fail;
    }
    _Py_uop_sym_apply_predicate_narrowing(ctx, ref, true);
    TEST_PREDICATE(!_Py_uop_sym_is_const(ctx, subject), "predicate narrowing incorrectly narrowed subject (inverted/true)");

    subject = _Py_uop_sym_new_unknown(ctx);
    value = _Py_uop_sym_new_const(ctx, one_obj);
    ref = _Py_uop_sym_new_predicate(ctx, subject, value, JIT_PRED_IS);
    if (PyJitRef_IsNull(subject) || PyJitRef_IsNull(value) || PyJitRef_IsNull(ref)) {
        goto fail;
    }
    TEST_PREDICATE(_Py_uop_sym_matches_type(ref, &PyBool_Type), "predicate is not boolean");
    TEST_PREDICATE(_Py_uop_sym_truthiness(ctx, ref) == -1, "predicate is not unknown");
    TEST_PREDICATE(_Py_uop_sym_is_const(ctx, ref) == false, "predicate is constant");
    TEST_PREDICATE(_Py_uop_sym_get_const(ctx, ref) == NULL, "predicate is not NULL");
    TEST_PREDICATE(_Py_uop_sym_is_const(ctx, value) == true, "value is not constant");
    TEST_PREDICATE(_Py_uop_sym_get_const(ctx, value) == one_obj, "value is not 1");
    _Py_uop_sym_set_const(ctx, ref, Py_False);
    TEST_PREDICATE(_Py_uop_sym_matches_type(ref, &PyBool_Type), "predicate is not boolean");
    TEST_PREDICATE(_Py_uop_sym_truthiness(ctx, ref) == 0, "predicate is not False");
    TEST_PREDICATE(_Py_uop_sym_is_const(ctx, ref) == true, "predicate is not constant");
    TEST_PREDICATE(_Py_uop_sym_get_const(ctx, ref) == Py_False, "predicate is not False");
    TEST_PREDICATE(_Py_uop_sym_is_const(ctx, value) == true, "value is not constant");
    TEST_PREDICATE(_Py_uop_sym_get_const(ctx, value) == one_obj, "value is not 1");

    val_big = PyNumber_Lshift(_PyLong_GetOne(), PyLong_FromLong(66));
    if (val_big == NULL) {
        goto fail;
    }

    JitOptRef ref_42 = _Py_uop_sym_new_const(ctx, val_42);
    JitOptRef ref_big = _Py_uop_sym_new_const(ctx, val_big);
    JitOptRef ref_int = _Py_uop_sym_new_compact_int(ctx);
    TEST_PREDICATE(_Py_uop_sym_is_compact_int(ref_42), "42 is not a compact int");
    TEST_PREDICATE(!_Py_uop_sym_is_compact_int(ref_big), "(1 << 66) is a compact int");
    TEST_PREDICATE(_Py_uop_sym_is_compact_int(ref_int), "compact int is not a compact int");
    TEST_PREDICATE(_Py_uop_sym_matches_type(ref_int, &PyLong_Type), "compact int is not an int");

    _Py_uop_sym_set_type(ctx, ref_int, &PyLong_Type);  // Should have no effect
    TEST_PREDICATE(_Py_uop_sym_is_compact_int(ref_int), "compact int is not a compact int after cast");
    TEST_PREDICATE(_Py_uop_sym_matches_type(ref_int, &PyLong_Type), "compact int is not an int after cast");

    _Py_uop_sym_set_type(ctx, ref_int, &PyFloat_Type);  // Should make it bottom
    TEST_PREDICATE(_Py_uop_sym_is_bottom(ref_int), "compact int cast to float isn't bottom");

    ref_int = _Py_uop_sym_new_compact_int(ctx);
    _Py_uop_sym_set_const(ctx, ref_int, val_43);
    TEST_PREDICATE(_Py_uop_sym_is_compact_int(ref_int), "43 is not a compact int");
    TEST_PREDICATE(_Py_uop_sym_matches_type(ref_int, &PyLong_Type), "43 is not an int");
    TEST_PREDICATE(_Py_uop_sym_get_const(ctx, ref_int) == val_43, "43 isn't 43");

    // Test recorded values

    /* Test that recorded values aren't treated as known values*/
    JitOptRef rv1 = _Py_uop_sym_new_unknown(ctx);
    _Py_uop_sym_set_recorded_value(ctx, rv1, val_42);
    TEST_PREDICATE(!_Py_uop_sym_matches_type(rv1, &PyLong_Type), "recorded value is treated as known");
    TEST_PREDICATE(_Py_uop_sym_get_const(ctx, rv1) == NULL, "recorded value is treated as known");
    TEST_PREDICATE(!_Py_uop_sym_is_compact_int(rv1), "recorded value is treated as known");

    /* Test that setting type or value narrows correctly */
    JitOptRef rv2 = _Py_uop_sym_new_unknown(ctx);
    _Py_uop_sym_set_recorded_value(ctx, rv2, val_42);
    _Py_uop_sym_set_const(ctx, rv2, val_42);
    TEST_PREDICATE(_Py_uop_sym_matches_type(rv2, &PyLong_Type), "recorded value doesn't narrow");
    TEST_PREDICATE(_Py_uop_sym_get_const(ctx, rv2) == val_42, "recorded value doesn't narrow");

    JitOptRef rv3 = _Py_uop_sym_new_unknown(ctx);
    _Py_uop_sym_set_recorded_value(ctx, rv3, val_42);
    _Py_uop_sym_set_type(ctx, rv3, &PyLong_Type);
    TEST_PREDICATE(_Py_uop_sym_matches_type(rv3, &PyLong_Type), "recorded value doesn't narrow");
    TEST_PREDICATE(_Py_uop_sym_get_const(ctx, rv3) == NULL, "recorded value with type is treated as known");

    JitOptRef rv4 = _Py_uop_sym_new_unknown(ctx);
    _Py_uop_sym_set_recorded_value(ctx, rv4, val_42);
    _Py_uop_sym_set_type_version(ctx, rv4, PyLong_Type.tp_version_tag);
    TEST_PREDICATE(_Py_uop_sym_matches_type(rv4, &PyLong_Type), "recorded value doesn't narrow");
    TEST_PREDICATE(_Py_uop_sym_get_const(ctx, rv4) == NULL, "recorded value with type is treated as known");

    // test recorded types

    /* Test that recorded type aren't treated as known values*/
    JitOptRef rt1 = _Py_uop_sym_new_unknown(ctx);
    _Py_uop_sym_set_recorded_type(ctx, rt1, &PyLong_Type);
    TEST_PREDICATE(!_Py_uop_sym_matches_type(rt1, &PyLong_Type), "recorded type is treated as known");
    TEST_PREDICATE(_Py_uop_sym_get_const(ctx, rt1) == NULL, "recorded type is treated as known value");

    /* Test that setting type or value narrows correctly */
    JitOptRef rt2 = _Py_uop_sym_new_unknown(ctx);
    _Py_uop_sym_set_recorded_type(ctx, rt2, &PyLong_Type);
    _Py_uop_sym_set_const(ctx, rt2, val_42);
    TEST_PREDICATE(_Py_uop_sym_matches_type(rt2, &PyLong_Type), "recorded value doesn't narrow");
    TEST_PREDICATE(_Py_uop_sym_get_const(ctx, rt2) == val_42, "recorded value doesn't narrow");

    JitOptRef rt3 = _Py_uop_sym_new_unknown(ctx);
    _Py_uop_sym_set_recorded_type(ctx, rt3, &PyLong_Type);
    _Py_uop_sym_set_type(ctx, rt3, &PyLong_Type);
    TEST_PREDICATE(_Py_uop_sym_matches_type(rt3, &PyLong_Type), "recorded value doesn't narrow");
    TEST_PREDICATE(_Py_uop_sym_get_const(ctx, rt3) == NULL, "known type is treated as known value");

    JitOptRef rt4 = _Py_uop_sym_new_unknown(ctx);
    _Py_uop_sym_set_recorded_type(ctx, rt4, &PyLong_Type);
    _Py_uop_sym_set_type_version(ctx, rt4, PyLong_Type.tp_version_tag);
    TEST_PREDICATE(_Py_uop_sym_matches_type(rt4, &PyLong_Type), "recorded value doesn't narrow");
    TEST_PREDICATE(_Py_uop_sym_get_const(ctx, rt4) == NULL, "recorded value with type is treated as known");

    // test recorded gen function

    PyObject *dict = PyDict_New();
    if (dict == NULL) {
        goto fail;
    }
    PyCodeObject *code = PyCode_NewEmpty(__FILE__, "uop_symbols_test", __LINE__);
    if (code == NULL) {
        goto fail;
    }
    func = (PyFunctionObject *)PyFunction_New((PyObject *)code, dict);
    if (func == NULL) {
        goto fail;
    }

    /* Test that recorded type aren't treated as known values*/
    JitOptRef rg1 = _Py_uop_sym_new_unknown(ctx);
    _Py_uop_sym_set_recorded_gen_func(ctx, rg1, func);
    TEST_PREDICATE(!_Py_uop_sym_matches_type(rg1, &PyGen_Type), "recorded gen func treated as generator");
    TEST_PREDICATE(_Py_uop_sym_get_probable_type(rg1) == &PyGen_Type, "recorded gen func not treated as generator");
    TEST_PREDICATE(_Py_uop_sym_get_const(ctx, rg1) == NULL, "recorded gen func is treated as known value");

    /* Test that setting type narrows correctly */

    JitOptRef rg2 = _Py_uop_sym_new_unknown(ctx);
    _Py_uop_sym_set_recorded_gen_func(ctx, rg2, func);
    _Py_uop_sym_set_type(ctx, rg2, &PyGen_Type);
    TEST_PREDICATE(!_Py_uop_sym_matches_type(rg2, &PyGen_Type), "recorded gen func treated as generator");
    TEST_PREDICATE(_Py_uop_sym_get_probable_type(rg2) == &PyGen_Type, "recorded gen func not treated as generator");
    TEST_PREDICATE(_Py_uop_sym_get_const(ctx, rg2) == NULL, "known type is treated as known value");

    JitOptRef rg3 = _Py_uop_sym_new_unknown(ctx);
    _Py_uop_sym_set_recorded_gen_func(ctx, rg3, func);
    _Py_uop_sym_set_type_version(ctx, rg3, PyGen_Type.tp_version_tag);
    TEST_PREDICATE(!_Py_uop_sym_matches_type(rg3, &PyGen_Type), "recorded gen func treated as generator");
    TEST_PREDICATE(_Py_uop_sym_get_probable_type(rg3) == &PyGen_Type, "recorded gen func not treated as generator");
    TEST_PREDICATE(_Py_uop_sym_get_const(ctx, rg3) == NULL, "recorded value with type is treated as known");

    /* Test contradictions */
    _Py_uop_sym_set_type(ctx, rv1, &PyFloat_Type);
     TEST_PREDICATE(_Py_uop_sym_is_bottom(rv1), "recorded value cast to other type isn't bottom");
    _Py_uop_sym_set_type_version(ctx, rv2, PyFloat_Type.tp_version_tag);
     TEST_PREDICATE(_Py_uop_sym_is_bottom(rv2), "recorded value cast to other type version isn't bottom");

    _Py_uop_sym_set_type(ctx, rt1, &PyFloat_Type);
     TEST_PREDICATE(_Py_uop_sym_is_bottom(rv1), "recorded type cast to other type isn't bottom");
    _Py_uop_sym_set_type_version(ctx, rt2, PyFloat_Type.tp_version_tag);
     TEST_PREDICATE(_Py_uop_sym_is_bottom(rv2), "recorded type cast to other type version isn't bottom");

    _Py_uop_sym_set_type(ctx, rg1, &PyFloat_Type);
     TEST_PREDICATE(_Py_uop_sym_is_bottom(rg1), "recorded gen func cast to other type isn't bottom");
    _Py_uop_sym_set_type_version(ctx, rg2, PyFloat_Type.tp_version_tag);
     TEST_PREDICATE(_Py_uop_sym_is_bottom(rg2), "recorded gen func cast to other type version isn't bottom");

    _Py_uop_abstractcontext_fini(ctx);
    Py_DECREF(val_42);
    Py_DECREF(val_43);
    Py_DECREF(val_big);
    Py_DECREF(tuple);
    Py_DECREF(func);
    Py_RETURN_NONE;

fail:
    _Py_uop_abstractcontext_fini(ctx);
    Py_XDECREF(val_42);
    Py_XDECREF(val_43);
    Py_XDECREF(val_big);
    Py_XDECREF(tuple);
    Py_XDECREF(func);
    return NULL;
}

PyObject *
_Py_uop_virtualization_test(PyObject *Py_UNUSED(self), PyObject *Py_UNUSED(ignored))
{
    JitOptContext context;
    JitOptContext *ctx = &context;
    _Py_uop_abstractcontext_init(ctx, NULL);

#if _Py_JIT_RAW_DOUBLE_SUPPORTED
    /* memcpy-based transport must preserve the exact IEEE-754 payload.  The
     * cases include both zero signs, infinity, and a quiet NaN payload. */
    static const uint64_t raw_double_cases[] = {
        UINT64_C(0x0000000000000000),
        UINT64_C(0x8000000000000000),
        UINT64_C(0x3ff8000000000000),
        UINT64_C(0x7ff0000000000000),
        UINT64_C(0x7ff8000000001234),
    };
    for (size_t i = 0; i < Py_ARRAY_LENGTH(raw_double_cases); i++) {
        uint64_t input_bits = raw_double_cases[i];
        double input;
        memcpy(&input, &input_bits, sizeof(input));

        _PyStackRef carrier = _PyJIT_PackRawDouble(input);
        uint64_t carrier_bits;
        memcpy(&carrier_bits, &carrier, sizeof(carrier_bits));
        TEST_PREDICATE(
            carrier_bits == input_bits,
            "packing a raw double changed its bit pattern"
        );

        double output = _PyJIT_UnpackRawDouble(carrier);
        uint64_t output_bits;
        memcpy(&output_bits, &output, sizeof(output_bits));
        TEST_PREDICATE(
            output_bits == input_bits,
            "unpacking a raw double changed its bit pattern"
        );

        _PyStackRef negated = _PyJIT_NegateRawDouble(carrier);
        uint64_t negated_bits;
        memcpy(&negated_bits, &negated, sizeof(negated_bits));
        TEST_PREDICATE(
            negated_bits == (input_bits ^ (UINT64_C(1) << 63)),
            "negating a raw double did not toggle exactly its sign bit"
        );
        _PyStackRef restored = _PyJIT_NegateRawDouble(negated);
        TEST_PREDICATE(
            memcmp(&restored, &carrier, sizeof(carrier)) == 0,
            "negating a raw double twice changed its bit pattern"
        );
    }
#endif

    /* Virtual binary arithmetic exposes a raw result plus two safe cleanup
     * values.  Its representation mask is an argument, and every generated
     * cache plan must select a form mapping back to the same base uop. */
    static const int virtual_binary_opcodes[] = {
        _BINARY_OP_ADD_FLOAT_VIRTUAL,
        _BINARY_OP_SUBTRACT_FLOAT_VIRTUAL,
        _BINARY_OP_MULTIPLY_FLOAT_VIRTUAL,
    };
    for (size_t i = 0; i < Py_ARRAY_LENGTH(virtual_binary_opcodes); i++) {
        int opcode = virtual_binary_opcodes[i];
        TEST_PREDICATE(
            _PyUop_Flags[opcode] == (HAS_ARG_FLAG | HAS_PURE_FLAG),
            "virtual float binary uop unexpectedly exits or escapes"
        );
        TEST_PREDICATE(
            _PyUop_num_popped(opcode, 0) == 2,
            "virtual float binary uop has the wrong input stack effect"
        );
        for (int incoming = 0; incoming <= MAX_CACHED_REGISTER; incoming++) {
            int depth = _PyUop_Caching[opcode].best[incoming];
            TEST_PREDICATE(
                depth >= 0 && depth <= MAX_CACHED_REGISTER,
                "virtual float binary uop has no usable cache form"
            );
            const _PyUopTOSentry *entry =
                &_PyUop_Caching[opcode].entries[depth];
            TEST_PREDICATE(
                entry->output == MAX_CACHED_REGISTER &&
                    entry->exit == depth,
                "virtual float binary uop has an invalid cache effect"
            );
            TEST_PREDICATE(
                _PyUop_Uncached[entry->opcode] == opcode,
                "virtual float binary cache form has an invalid base uop"
            );
        }
    }

    /* Float comparisons use the ordinary COMPARE_OP argument in bits 0-7,
     * plus two representation bits above it.  The uop always emits a bool and
     * two cleanup values; a raw input's cleanup value is NULL so the following
     * decref operation is optimized away without touching raw bits. */
    TEST_PREDICATE(
        (_Py_JIT_COMPARE_VIRTUAL_FLOAT_RAW_MASK & UINT8_MAX) == 0,
        "virtual-float comparison representation bits overlap COMPARE_OP"
    );
    TEST_PREDICATE(
        _PyUop_Flags[_COMPARE_OP_FLOAT_VIRTUAL] ==
            (HAS_ARG_FLAG | HAS_PURE_FLAG),
        "virtual-float comparison unexpectedly exits or escapes"
    );
    TEST_PREDICATE(
        _PyUop_num_popped(_COMPARE_OP_FLOAT_VIRTUAL, 0) == 2,
        "virtual-float comparison has the wrong input stack effect"
    );
    for (int incoming = 0; incoming <= MAX_CACHED_REGISTER; incoming++) {
        int depth = _PyUop_Caching[_COMPARE_OP_FLOAT_VIRTUAL].best[incoming];
        TEST_PREDICATE(
            depth == 2,
            "virtual-float comparison selected an invalid cache depth"
        );
        const _PyUopTOSentry *entry =
            &_PyUop_Caching[_COMPARE_OP_FLOAT_VIRTUAL].entries[depth];
        TEST_PREDICATE(
            entry->output == 3 && entry->exit == depth &&
                entry->opcode == _COMPARE_OP_FLOAT_VIRTUAL_r23,
            "virtual-float comparison has an invalid register-cache form"
        );
        TEST_PREDICATE(
            _PyUop_Uncached[entry->opcode] == _COMPARE_OP_FLOAT_VIRTUAL,
            "virtual-float comparison cache form has an invalid base uop"
        );
    }

    /* Truth testing replaces one raw value with one bool.  Unlike
     * materialization it cannot fail, so cached references below the raw value
     * stay cached and every incoming depth has its own generated form. */
    static const int truth_outputs[MAX_CACHED_REGISTER + 1] = {1, 1, 2, 3};
    static const int truth_opcodes[MAX_CACHED_REGISTER + 1] = {
        _TO_BOOL_FLOAT_VIRTUAL_r01,
        _TO_BOOL_FLOAT_VIRTUAL_r11,
        _TO_BOOL_FLOAT_VIRTUAL_r22,
        _TO_BOOL_FLOAT_VIRTUAL_r33,
    };
    TEST_PREDICATE(
        _PyUop_Flags[_TO_BOOL_FLOAT_VIRTUAL] == HAS_PURE_FLAG,
        "virtual-float truth test unexpectedly exits or escapes"
    );
    TEST_PREDICATE(
        _PyUop_num_popped(_TO_BOOL_FLOAT_VIRTUAL, 0) == 1,
        "virtual-float truth test has the wrong input stack effect"
    );
    for (int depth = 0; depth <= MAX_CACHED_REGISTER; depth++) {
        const _PyUopTOSentry *entry =
            &_PyUop_Caching[_TO_BOOL_FLOAT_VIRTUAL].entries[depth];
        TEST_PREDICATE(
            entry->output == truth_outputs[depth] &&
                entry->exit == depth &&
                entry->opcode == truth_opcodes[depth],
            "virtual-float truth test has an invalid register-cache form"
        );
        TEST_PREDICATE(
            _PyUop_Uncached[entry->opcode] == _TO_BOOL_FLOAT_VIRTUAL,
            "virtual-float truth cache form has an invalid base uop"
        );
    }

    TEST_PREDICATE(
        _PyUop_Flags[_UNARY_NEGATIVE_FLOAT_VIRTUAL] ==
            (HAS_ARG_FLAG | HAS_PURE_FLAG),
        "virtual float negation unexpectedly exits or escapes"
    );
    TEST_PREDICATE(
        _PyUop_num_popped(_UNARY_NEGATIVE_FLOAT_VIRTUAL, 0) == 1,
        "virtual float negation has the wrong input stack effect"
    );
    static const int virtual_unary_outputs[MAX_CACHED_REGISTER] = {2, 2, 3};
    for (int incoming = 0; incoming <= MAX_CACHED_REGISTER; incoming++) {
        int depth = _PyUop_Caching[_UNARY_NEGATIVE_FLOAT_VIRTUAL].best[incoming];
        TEST_PREDICATE(
            depth >= 0 && depth < MAX_CACHED_REGISTER,
            "virtual float negation has no usable cache form"
        );
        const _PyUopTOSentry *entry =
            &_PyUop_Caching[_UNARY_NEGATIVE_FLOAT_VIRTUAL].entries[depth];
        TEST_PREDICATE(
            entry->output == virtual_unary_outputs[depth] &&
                entry->exit == depth,
            "virtual float negation has an invalid cache effect"
        );
        TEST_PREDICATE(
            _PyUop_Uncached[entry->opcode] ==
                _UNARY_NEGATIVE_FLOAT_VIRTUAL,
            "virtual float negation cache form has an invalid base uop"
        );
    }

    /* A one-input pop has one generated form for each possible incoming cache
     * depth.  Validate that all forms reduce the cache depth by one when the
     * value is cached, and all map back to the raw discard uop. */
    static const int discard_outputs[MAX_CACHED_REGISTER + 1] = {0, 0, 1, 2};
    static const int discard_opcodes[MAX_CACHED_REGISTER + 1] = {
        _POP_TOP_VIRTUAL_FLOAT_r00,
        _POP_TOP_VIRTUAL_FLOAT_r10,
        _POP_TOP_VIRTUAL_FLOAT_r21,
        _POP_TOP_VIRTUAL_FLOAT_r32,
    };
    TEST_PREDICATE(
        _PyUop_Flags[_POP_TOP_VIRTUAL_FLOAT] == 0,
        "virtual-float discard unexpectedly exits or escapes"
    );
    TEST_PREDICATE(
        _PyUop_num_popped(_POP_TOP_VIRTUAL_FLOAT, 0) == 1,
        "virtual-float discard has the wrong stack effect"
    );
    for (int depth = 0; depth <= MAX_CACHED_REGISTER; depth++) {
        const _PyUopTOSentry *entry =
            &_PyUop_Caching[_POP_TOP_VIRTUAL_FLOAT].entries[depth];
        TEST_PREDICATE(
            entry->output == discard_outputs[depth] &&
                entry->exit == depth &&
                entry->opcode == discard_opcodes[depth],
            "virtual-float discard has an invalid register-cache form"
        );
        TEST_PREDICATE(
            _PyUop_Uncached[entry->opcode] == _POP_TOP_VIRTUAL_FLOAT,
            "virtual-float discard cache form has an invalid base uop"
        );
    }

    /* Materialization replaces one raw input with one boxed output.  It may
     * fail, but it must use the pop-on-error convention: generated error paths
     * spill only the other cached Python references, never the raw input. */
    static const int materialize_outputs[MAX_CACHED_REGISTER + 1] = {1, 1, 2, 3};
    static const int materialize_opcodes[MAX_CACHED_REGISTER + 1] = {
        _MATERIALIZE_VIRTUAL_FLOAT_r01,
        _MATERIALIZE_VIRTUAL_FLOAT_r11,
        _MATERIALIZE_VIRTUAL_FLOAT_r22,
        _MATERIALIZE_VIRTUAL_FLOAT_r33,
    };
    TEST_PREDICATE(
        _PyUop_Flags[_MATERIALIZE_VIRTUAL_FLOAT] == HAS_ERROR_FLAG,
        "virtual-float materialization has invalid error semantics"
    );
    TEST_PREDICATE(
        _PyUop_num_popped(_MATERIALIZE_VIRTUAL_FLOAT, 0) == 1,
        "virtual-float materialization has the wrong input stack effect"
    );
    for (int depth = 0; depth <= MAX_CACHED_REGISTER; depth++) {
        const _PyUopTOSentry *entry =
            &_PyUop_Caching[_MATERIALIZE_VIRTUAL_FLOAT].entries[depth];
        TEST_PREDICATE(
            entry->output == materialize_outputs[depth] &&
                entry->exit == depth &&
                entry->opcode == materialize_opcodes[depth],
            "virtual-float materialization has an invalid register-cache form"
        );
        TEST_PREDICATE(
            _PyUop_Uncached[entry->opcode] == _MATERIALIZE_VIRTUAL_FLOAT,
            "virtual-float materialization cache form has an invalid base uop"
        );
    }

    JitOptRef virtual_float = _Py_uop_sym_new_type(ctx, &PyFloat_Type);
    TEST_PREDICATE(
        _Py_uop_alloc_get(ctx, virtual_float).state == JIT_ALLOC_CONCRETE,
        "new symbol is not concrete"
    );
    JitVirtualRef virtual_ref =
        _Py_uop_alloc_new_virtual_float(ctx, virtual_float);
    TEST_PREDICATE(
        virtual_ref != JIT_VIRTUAL_REF_NULL,
        "failed to create virtual float"
    );
    TEST_PREDICATE(
        _Py_uop_alloc_is_virtual(ctx, virtual_float),
        "virtual float is concrete"
    );
    TEST_PREDICATE(
        _Py_uop_alloc_get(
            ctx, PyJitRef_Borrow(virtual_float)).virtual_ref == virtual_ref,
        "borrowed reference has different allocation state"
    );
    TEST_PREDICATE(
        _Py_uop_alloc_get(
            ctx, PyJitRef_MakeUnique(virtual_float)).virtual_ref == virtual_ref,
        "unique reference has different allocation state"
    );
    TEST_PREDICATE(
        _Py_uop_alloc_new_virtual_float(ctx, virtual_float) == virtual_ref,
        "virtualizing a symbol twice changed its identity"
    );
    const JitVirtualObject *virtual_object =
        _Py_uop_alloc_resolve_virtual(ctx, virtual_float);
    TEST_PREDICATE(
        virtual_object != NULL && virtual_object->kind == JIT_VIRTUAL_FLOAT,
        "virtual float did not resolve to a float descriptor"
    );
    TEST_PREDICATE(
        _Py_uop_sym_matches_type(virtual_float, &PyFloat_Type),
        "virtual float lost its ordinary float type"
    );
    TEST_PREDICATE(
        _Py_uop_sym_is_not_null(virtual_float),
        "virtual float lost its ordinary non-null fact"
    );
    JitOptRef materialized = _Py_uop_alloc_materialize_virtual_float(
        ctx, PyJitRef_MakeUnique(virtual_float));
    TEST_PREDICATE(
        !PyJitRef_IsNull(materialized) && PyJitRef_IsUnique(materialized),
        "materialization did not produce a fresh unique symbol"
    );
    TEST_PREDICATE(
        _Py_uop_sym_matches_type(materialized, &PyFloat_Type) &&
            !_Py_uop_alloc_is_virtual(ctx, materialized),
        "materialization did not produce a concrete exact float"
    );
    TEST_PREDICATE(
        _Py_uop_alloc_is_virtual(ctx, virtual_float),
        "materialization retroactively changed the virtual symbol"
    );
    TEST_PREDICATE(
        PyJitRef_IsNull(
            _Py_uop_alloc_materialize_virtual_float(ctx, virtual_float)),
        "materialization accepted a non-unique virtual value"
    );

    JitOptSymbol foreign_symbol = {.tag = JIT_SYM_UNKNOWN_TAG};
    TEST_PREDICATE(
        !_Py_uop_alloc_is_virtual(ctx, PyJitRef_NULL),
        "NULL reference is virtual"
    );
    TEST_PREDICATE(
        !_Py_uop_alloc_is_virtual(ctx, PyJitRef_WrapInvalid(ctx)),
        "invalid reference is virtual"
    );
    TEST_PREDICATE(
        _Py_uop_alloc_new_virtual_float(
            ctx, PyJitRef_Wrap(&foreign_symbol)) == JIT_VIRTUAL_REF_NULL,
        "out-of-arena reference was virtualized"
    );
    TEST_PREDICATE(
        !_Py_uop_alloc_set_concrete(ctx, PyJitRef_NULL),
        "NULL reference was made concrete"
    );
    JitOptRef concrete_int = _Py_uop_sym_new_type(ctx, &PyLong_Type);
    TEST_PREDICATE(
        _Py_uop_alloc_new_virtual_float(ctx, concrete_int) ==
            JIT_VIRTUAL_REF_NULL,
        "non-float symbol was virtualized as a float"
    );
    TEST_PREDICATE(
        !_Py_uop_alloc_is_virtual(ctx, concrete_int),
        "rejected non-float symbol became virtual"
    );

    /* The generated optimizer moves JitOptRefs through locals and the stack.
     * Since allocation state is keyed by the unwrapped symbol, these ordinary
     * assignments are the allocation-domain transfer functions for loads and
     * swaps; they do not need a parallel representation stack. */
    _Py_UOpsAbstractFrame *frame = &ctx->frames[0];
    frame->locals = ctx->locals_array;
    frame->locals_len = 2;
    frame->stack = ctx->stack_array;
    frame->stack_len = 4;
    frame->locals[0] = PyJitRef_RemoveUnique(
        PyJitRef_MakeUnique(virtual_float));
    frame->locals[1] = concrete_int;

    JitOptRef *stack_pointer = frame->stack;
    *stack_pointer++ = frame->locals[0];  // _LOAD_FAST
    *stack_pointer++ = PyJitRef_Borrow(frame->locals[0]);
    JitOptRef ordinary_float = _Py_uop_sym_new_type(ctx, &PyFloat_Type);
    *stack_pointer++ = ordinary_float;
    TEST_PREDICATE(
        _Py_uop_alloc_validate_frame(ctx, frame, stack_pointer),
        "allocation state is invalid after loading values"
    );
    ctx->virtuals.arena[virtual_ref - 1].kind = JIT_VIRTUAL_INVALID;
    TEST_PREDICATE(
        !_Py_uop_alloc_validate_frame(ctx, frame, stack_pointer),
        "invalid virtual descriptor passed frame validation"
    );
    ctx->virtuals.arena[virtual_ref - 1].kind = JIT_VIRTUAL_FLOAT;
    TEST_PREDICATE(
        _Py_uop_alloc_frame_has_virtual(ctx, frame, stack_pointer),
        "loaded virtual value is not live"
    );

    /* Context-wide scans include suspended callers as well as the current
     * inlined frame.  The current frame's local stack pointer is supplied by
     * the abstract-interpreter loop; callers use their saved pointer. */
    frame->stack_pointer = stack_pointer;
    _Py_UOpsAbstractFrame *inner = &ctx->frames[1];
    inner->locals = &ctx->locals_array[2];
    inner->locals_len = 1;
    inner->locals[0] = concrete_int;
    inner->stack = &ctx->stack_array[4];
    inner->stack_len = 1;
    inner->stack_pointer = inner->stack;
    ctx->curr_frame_depth = 2;
    ctx->frame = inner;
    TEST_PREDICATE(
        _Py_uop_alloc_context_virtual_count(ctx, inner->stack_pointer) == 3,
        "context scan missed virtual values in a caller"
    );
    TEST_PREDICATE(
        _Py_uop_alloc_context_has_virtual(ctx, inner->stack_pointer),
        "context scan treated a virtual caller as concrete"
    );
    TEST_PREDICATE(
        _Py_uop_alloc_validate_context(ctx, inner->stack_pointer),
        "valid allocation state failed whole-context validation"
    );
    ctx->curr_frame_depth = 1;
    ctx->frame = frame;
    TEST_PREDICATE(
        !_Py_uop_alloc_is_virtual(ctx, ordinary_float),
        "ordinary producer created a virtual value"
    );

    JitOptRef temp = frame->stack[0];  // _SWAP
    frame->stack[0] = frame->stack[2];
    frame->stack[2] = temp;
    TEST_PREDICATE(
        !_Py_uop_alloc_is_virtual(ctx, frame->stack[0]) &&
            _Py_uop_alloc_is_virtual(ctx, frame->stack[2]),
        "swap did not move allocation state with its symbols"
    );

    stack_pointer--;  // _POP_TOP makes the top virtual value dead.
    TEST_PREDICATE(
        _Py_uop_alloc_validate_frame(ctx, frame, stack_pointer),
        "allocation state is invalid after popping a value"
    );
    stack_pointer--;
    frame->locals[0] = concrete_int;
    TEST_PREDICATE(
        !_Py_uop_alloc_frame_has_virtual(ctx, frame, stack_pointer),
        "dead virtual values are still reported as live"
    );

    TEST_PREDICATE(
        _Py_uop_alloc_set_concrete(ctx, virtual_float),
        "failed to make a valid reference concrete"
    );
    TEST_PREDICATE(
        !_Py_uop_alloc_is_virtual(ctx, virtual_float),
        "reference remained virtual after being made concrete"
    );

    virtual_float = _Py_uop_sym_new_type(ctx, &PyFloat_Type);
    ctx->virtuals.virtual_max_number = ctx->virtuals.virtual_curr_number;
    TEST_PREDICATE(
        _Py_uop_alloc_new_virtual_float(ctx, virtual_float) ==
            JIT_VIRTUAL_REF_NULL,
        "virtual arena exhaustion created a descriptor"
    );
    TEST_PREDICATE(
        ctx->done && ctx->out_of_space && ctx->retry_without_virtuals,
        "virtual arena exhaustion did not request a non-virtual retry"
    );

    _Py_uop_abstractcontext_fini(ctx);
    Py_RETURN_NONE;

fail:
    _Py_uop_abstractcontext_fini(ctx);
    return NULL;
}

#endif /* _Py_TIER2 */
