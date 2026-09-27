#ifdef _Py_TIER2

/*
 * This file contains the support code for CPython's uops optimizer.
 * It also performs some simple optimizations.
 * It performs a traditional data-flow analysis[1] over the trace of uops.
 * Using the information gained, it chooses to emit, or skip certain instructions
 * if possible.
 *
 * [1] For information on data-flow analysis, please see
 * https://clang.llvm.org/docs/DataFlowAnalysisIntro.html
 *
 * */
#include "Python.h"
#include "opcode.h"
#include "pycore_call.h"          // _Py_METH_CALL_FLAGS
#include "pycore_dict.h"
#include "pycore_interp.h"
#include "pycore_opcode_metadata.h"
#include "pycore_opcode_utils.h"
#include "pycore_pystate.h"       // _PyInterpreterState_GET()
#include "pycore_pyatomic_ft_wrappers.h" // FT_ATOMIC_*
#include "pycore_tstate.h"        // _PyThreadStateImpl
#include "pycore_uop_metadata.h"
#include "pycore_long.h"
#include "pycore_interpframe.h"  // _PyFrame_GetCode
#include "pycore_jit.h"          // _Py_JIT_RAW_DOUBLE_SUPPORTED
#include "pycore_optimizer.h"
#include "pycore_object.h"
#include "pycore_function.h"
#include "pycore_uop_ids.h"
#include "pycore_range.h"
#include "pycore_unicodeobject.h"
#include "pycore_ceval.h"
#include "pycore_floatobject.h"
#include "pycore_setobject.h"
#include "pycore_typeobject.h"

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

#ifdef Py_DEBUG
    extern const char *_PyUOpName(int index);
    extern void _PyUOpPrint(const _PyUOpInstruction *uop);
    extern void _PyUOpSymPrint(JitOptContext *ctx, JitOptRef ref);
    static const char *const DEBUG_ENV = "PYTHON_OPT_DEBUG";
    static inline int get_lltrace(void) {
        char *uop_debug = Py_GETENV(DEBUG_ENV);
        int lltrace = 0;
        if (uop_debug != NULL && *uop_debug >= '0') {
            lltrace = *uop_debug - '0';  // TODO: Parse an int and all that
        }
        return lltrace;
    }
    #define DPRINTF(level, ...) \
    if (get_lltrace() >= (level)) { printf(__VA_ARGS__); }

static void
dump_abstract_stack(
    JitOptContext *ctx,
    _Py_UOpsAbstractFrame *frame,
    JitOptRef *stack_pointer)
{
    printf("    locals=[");
    for (int i = 0 ; i < frame->locals_len; i++) {
        if (i > 0) {
            printf(", ");
        }
        _PyUOpSymPrint(ctx, frame->locals[i]);
    }
    printf("]\n");
    if (stack_pointer < frame->stack) {
        printf("    stack=%d\n", (int)(stack_pointer - frame->stack));
    }
    else {
        printf("    stack=[");
        for (JitOptRef *ptr = frame->stack; ptr < stack_pointer; ptr++) {
            if (ptr != frame->stack) {
                printf(", ");
            }
            _PyUOpSymPrint(ctx, *ptr);
        }
        printf("]\n");
    }
    fflush(stdout);
}

static void
dump_uop(JitOptContext *ctx, const char *label, int index,
              const _PyUOpInstruction *instr, JitOptRef *stack_pointer)
{
    if (get_lltrace() >= 3) {
        printf("%4d %s: ", index, label);
        _PyUOpPrint(instr);
        printf("\n");
        if (get_lltrace() >= 5 && ctx->frame->code != ((PyCodeObject *)&_Py_InitCleanup)) {
            dump_abstract_stack(ctx, ctx->frame, stack_pointer);
        }
    }
}

static void
dump_uops(JitOptContext *ctx, const char *label,
          _PyUOpInstruction *start, JitOptRef *stack_pointer)
{
    int current_len = uop_buffer_length(&ctx->out_buffer);
    int added_count = (int)(ctx->out_buffer.next - start);
    for (int j = 0; j < added_count; j++) {
        dump_uop(ctx, label, current_len - added_count + j, &start[j], stack_pointer);
    }
}

#define DUMP_UOP dump_uop
#define DUMP_UOPS dump_uops

#else
    #define DPRINTF(level, ...)
    #define DUMP_UOP(ctx, label, index, instr, stack_pointer)
    #define DUMP_UOPS(ctx, label, start, stack_pointer)
#endif

static int
get_mutations(PyObject* dict) {
    assert(PyDict_CheckExact(dict));
    PyDictObject *d = (PyDictObject *)dict;
    uint64_t tag = FT_ATOMIC_LOAD_UINT64_RELAXED(d->_ma_watcher_tag);
    return (tag >> DICT_MAX_WATCHERS) & ((1 << DICT_WATCHED_MUTATION_BITS) - 1);
}

static void
increment_mutations(PyObject* dict) {
    assert(PyDict_CheckExact(dict));
    PyDictObject *d = (PyDictObject *)dict;
    FT_ATOMIC_ADD_UINT64(d->_ma_watcher_tag, 1ULL << DICT_MAX_WATCHERS);
}

/* The first two dict watcher IDs are reserved for CPython,
 * so we don't need to check that they haven't been used */
#define BUILTINS_WATCHER_ID 0
#define GLOBALS_WATCHER_ID  1
#define TYPE_WATCHER_ID  0

static int
globals_watcher_callback(PyDict_WatchEvent event, PyObject* dict,
                         PyObject* key, PyObject* new_value)
{
    RARE_EVENT_STAT_INC(watched_globals_modification);
    assert(get_mutations(dict) < _Py_MAX_ALLOWED_GLOBALS_MODIFICATIONS);
    _Py_Executors_InvalidateDependency(_PyInterpreterState_GET(), dict, 1);
    increment_mutations(dict);
    PyDict_Unwatch(GLOBALS_WATCHER_ID, dict);
    return 0;
}

static int
type_watcher_callback(PyTypeObject* type)
{
    _Py_Executors_InvalidateDependency(_PyInterpreterState_GET(), type, 1);
    PyType_Unwatch(TYPE_WATCHER_ID, (PyObject *)type);
    return 0;
}

static int
_setup_optimizer_watchers(void *Py_UNUSED(arg))
{
    PyInterpreterState *interp = _PyInterpreterState_GET();
    FT_ATOMIC_STORE_PTR_RELEASE(
        interp->dict_state.watchers[GLOBALS_WATCHER_ID],
        globals_watcher_callback);
    interp->type_watchers[TYPE_WATCHER_ID] = type_watcher_callback;
    return 0;
}

static void
watch_type(PyTypeObject *type, _PyBloomFilter *filter)
{
    if (_Py_IsImmortal(type) && (type->tp_flags & Py_TPFLAGS_IMMUTABLETYPE)) {
        return;
    }
    PyType_Watch(TYPE_WATCHER_ID, (PyObject *)type);
    _Py_BloomFilter_Add(filter, type);
}

static PyObject *
convert_global_to_const(_PyUOpInstruction *inst, PyObject *obj)
{
    assert(inst->opcode == _LOAD_GLOBAL_MODULE || inst->opcode == _LOAD_GLOBAL_BUILTINS || inst->opcode == _LOAD_ATTR_MODULE);
    assert(PyDict_CheckExact(obj));
    PyDictObject *dict = (PyDictObject *)obj;
    assert(dict->ma_keys->dk_kind == DICT_KEYS_UNICODE);
    PyDictUnicodeEntry *entries = DK_UNICODE_ENTRIES(dict->ma_keys);
    int64_t index = inst->operand1;
    assert(index <= UINT16_MAX);
    if ((int)index >= dict->ma_keys->dk_nentries) {
        return NULL;
    }
    PyDictKeysObject *keys = dict->ma_keys;
    if (keys->dk_version != inst->operand0) {
        return NULL;
    }
    PyObject *res = entries[index].me_value;
    if (res == NULL) {
        return NULL;
    }
    if (_Py_IsImmortal(res)) {
        inst->opcode = _LOAD_CONST_INLINE_BORROW;
    } else {
        inst->opcode = _LOAD_CONST_INLINE;
    }
    inst->operand0 = (uint64_t)res;
    return res;
}

static bool
incorrect_keys(PyObject *obj, uint32_t version)
{
    if (!PyDict_CheckExact(obj)) {
        return true;
    }
    PyDictObject *dict = (PyDictObject *)obj;
    return dict->ma_keys->dk_version != version;
}


#define STACK_LEVEL()     ((int)(stack_pointer - ctx->frame->stack))
#define STACK_SIZE()      ((int)(ctx->frame->stack_len))

static inline int
is_terminator_uop(const _PyUOpInstruction *uop)
{
    int opcode = uop->opcode;
    return (
        opcode == _EXIT_TRACE ||
        opcode == _JUMP_TO_TOP ||
        opcode == _DYNAMIC_EXIT ||
        opcode == _DEOPT
    );
}

/* Any uop with one of these effects may expose the frame, branch to Tier 1,
 * or otherwise require its operands to be real Python references. */
#define VIRTUAL_BARRIER_FLAGS \
    (HAS_EXIT_FLAG | HAS_DEOPT_FLAG | HAS_ERROR_FLAG | \
     HAS_ESCAPES_FLAG | HAS_SYNC_SP_FLAG)

/* Internal result distinct from ordinary "not ready": rerun the complete
 * abstract interpretation pass with only boxed representations. */
#define OPTIMIZE_RETRY_WITHOUT_VIRTUALS (-2)

static bool
is_virtualization_barrier(const _PyUOpInstruction *uop)
{
    /* Centralize the conservative boundary between raw-only Tier-2 values and
     * operations that can observe normal Python object/frame semantics. */
    /* These consumers require a concrete Python object even though their
     * metadata does not describe them as exits or safepoints. */
    switch (uop->opcode) {
        case _CHECK_VALIDITY:
            /* A raw value can only reach this check through pure virtual uops.
             * No operation capable of invalidating the executor has run, so
             * remove_unneeded_uops() will discard the redundant check. */
            return false;
        case _COPY:
        case _IS_OP:
        case _MAKE_HEAP_SAFE:
        case _PUSH_FRAME:
            return true;
        default:
            break;
    }
    return is_terminator_uop(uop) ||
        (_PyUop_Flags[uop->opcode] & VIRTUAL_BARRIER_FLAGS) != 0;
}

static bool
virtuals_cross_unsafe_barrier(
    const _PyUOpInstruction *start,
    const _PyUOpInstruction *end,
    int live_virtuals_before,
    bool live_virtuals_after)
{
    /* Audit the exact output slice produced for one input uop.  This catches
     * both copied source barriers and replacement sequences emitted by an
     * optimizer transfer rule. */
    if (live_virtuals_before < 0) {
        return true;
    }
    bool raw_is_live = live_virtuals_before > 0;
    bool materialized = false;
    for (const _PyUOpInstruction *uop = start; uop < end; uop++) {
        if (uop->opcode == _MATERIALIZE_VIRTUAL_FLOAT) {
            /* This allocation's error path is valid only when it consumes the
             * sole raw value in all active frames. */
            if (!raw_is_live || live_virtuals_before != 1 || materialized) {
                return true;
            }
            raw_is_live = false;
            materialized = true;
            continue;
        }
        if (uop->opcode == _SWAP_FAST_VIRTUAL_FLOAT) {
            /* Its uncommon old-object path reconstructs every existing
             * shadow before allocating or running a destructor; the common
             * path only moves one raw value from the stack into the shadow
             * table.  The uop therefore owns its escape discipline. */
            continue;
        }
        if (uop->opcode == _BINARY_OP_POWER_FLOAT_VIRTUAL_RAW_CONST) {
            /* It consumes its raw/borrowed inputs before the only error edge
             * and produces another raw float only on success. */
            continue;
        }
        if (uop->opcode == _GUARD_NOS_FLOAT_VIRTUAL_FAST) {
            /* Its hot path only checks the boxed NOS operand.  Its cold path
             * reconstructs one identity-preserving box in both the shadowed
             * local and TOS before either its error or side exit. */
            continue;
        }
        if (raw_is_live && is_virtualization_barrier(uop)) {
            return true;
        }
    }
    /* The abstract transfer for materialization must agree that no raw value
     * remains before a later barrier or error path can expose the frame. */
    return materialized && live_virtuals_after;
}

#define CURRENT_FRAME_IS_INIT_SHIM() (ctx->frame->code == ((PyCodeObject *)&_Py_InitCleanup))

#define GETLOCAL(idx)          ((ctx->frame->locals[idx]))

#define REPLACE_OP(INST, OP, ARG, OPERAND)    \
    (INST)->opcode = OP;            \
    (INST)->oparg = ARG;            \
    (INST)->operand0 = OPERAND;

#define ADD_OP(OP, ARG, OPERAND) add_op(ctx, this_instr, (OP), (ARG), (OPERAND))

static inline void
add_op(JitOptContext *ctx, _PyUOpInstruction *this_instr,
       uint16_t opcode, uint16_t oparg, uintptr_t operand0)
{
    _PyUOpInstruction *out = ctx->out_buffer.next;
    assert(out < ctx->out_buffer.end);
    out->opcode = (opcode);
    out->format = this_instr->format;
    out->oparg = (oparg);
    out->target = this_instr->target;
    out->operand0 = (operand0);
    out->operand1 = this_instr->operand1;
#ifdef Py_STATS
    out->fitness = this_instr->fitness;
#endif
    ctx->out_buffer.next++;
}

static bool
uop_handles_virtual_float(const _PyUOpInstruction *uop)
{
    /* These uops either consume raw double bits directly or contain their own
     * precise materialization rule.  All other barriers must see only normal
     * PyStackRefs. */
    if (uop->opcode == _BINARY_OP) {
        return uop->oparg == NB_POWER || uop->oparg == NB_INPLACE_POWER;
    }
    switch (uop->opcode) {
        case _GUARD_TOS_FLOAT:
        case _GUARD_NOS_FLOAT:
        case _GUARD_NOS_FLOAT_VIRTUAL_FAST:
        case _BINARY_OP_ADD_FLOAT:
        case _BINARY_OP_SUBTRACT_FLOAT:
        case _BINARY_OP_MULTIPLY_FLOAT:
        case _BINARY_OP_ADD_FLOAT_VIRTUAL_BOXED_BOXED:
        case _BINARY_OP_ADD_FLOAT_VIRTUAL_RAW_BOXED:
        case _BINARY_OP_ADD_FLOAT_VIRTUAL_BOXED_RAW:
        case _BINARY_OP_ADD_FLOAT_VIRTUAL_RAW_RAW:
        case _BINARY_OP_ADD_FLOAT_INPLACE_BOXED_RAW:
        case _BINARY_OP_ADD_FLOAT_INPLACE_RAW_BOXED:
        case _BINARY_OP_SUBTRACT_FLOAT_VIRTUAL:
        case _BINARY_OP_SUBTRACT_FLOAT_INPLACE_BOXED_RAW:
        case _BINARY_OP_SUBTRACT_FLOAT_INPLACE_RAW_BOXED:
        case _BINARY_OP_MULTIPLY_FLOAT_VIRTUAL_BOXED_BOXED:
        case _BINARY_OP_MULTIPLY_FLOAT_VIRTUAL_RAW_BOXED:
        case _BINARY_OP_MULTIPLY_FLOAT_VIRTUAL_BOXED_RAW:
        case _BINARY_OP_MULTIPLY_FLOAT_VIRTUAL_RAW_RAW:
        case _BINARY_OP_MULTIPLY_FLOAT_INPLACE_BOXED_RAW:
        case _BINARY_OP_MULTIPLY_FLOAT_INPLACE_RAW_BOXED:
        case _BINARY_OP_POWER_FLOAT_VIRTUAL_RAW_CONST:
        case _UNARY_NEGATIVE:
        case _UNARY_NEGATIVE_FLOAT_VIRTUAL:
        case _TO_BOOL:
        case _TO_BOOL_FLOAT_VIRTUAL:
        case _COMPARE_OP_FLOAT:
        case _COMPARE_OP_FLOAT_VIRTUAL:
        case _POP_TOP:
        case _POP_TOP_OPARG:
        case _POP_TOP_VIRTUAL_FLOAT:
        case _MATERIALIZE_VIRTUAL_FLOAT:
        case _MATERIALIZE_VIRTUAL_FAST_LOCAL:
        case _LOAD_FAST_VIRTUAL_FLOAT:
        case _LOAD_FAST_AND_CLEAR_VIRTUAL_FLOAT:
        case _SWAP_FAST_VIRTUAL_FLOAT:
        case _SWAP_FAST:
            return true;
        default:
            return false;
    }
}

static bool
uop_preserves_virtual_fast_locals(const _PyUOpInstruction *uop)
{
    /* These source uops either cannot invoke Python on their hot path or are
     * rewritten to a guarded form with that property.  Their side exits use
     * normal Tier-2 shadow-local reconstruction. */
    switch (uop->opcode) {
        case _BINARY_OP_SUBSCR_LIST_INT:
        case _STORE_SUBSCR_LIST_INT:
        case _STORE_SUBSCR_LIST_INT_FLOAT:
            return true;
        default:
            return false;
    }
}

static int
find_unique_virtual_fast_local_alias(JitOptContext *ctx, JitOptRef value)
{
    /* A failing type guard may reconstruct a raw operand without changing
     * Python identity only when that operand aliases exactly one root-frame
     * shadow local.  The specialized guard installs one box in that local and
     * duplicates the same reference onto the operand stack. */
    if (ctx->curr_frame_depth != 1) {
        return -1;
    }
    int match = -1;
    int limit = ctx->frame->locals_len;
    if (limit > _PY_JIT_MAX_VIRTUAL_FAST_LOCALS) {
        limit = _PY_JIT_MAX_VIRTUAL_FAST_LOCALS;
    }
    for (int local = 0; local < limit; local++) {
        JitOptRef candidate = ctx->frame->locals[local];
        if (_Py_uop_alloc_is_virtual(ctx, candidate) &&
            PyJitRef_Unwrap(candidate) == PyJitRef_Unwrap(value))
        {
            if (match >= 0) {
                return -1;
            }
            match = local;
        }
    }
    return match;
}

static bool
materialize_single_stack_virtual(
    JitOptContext *ctx,
    _PyUOpInstruction *this_instr,
    JitOptRef *stack_pointer)
{
    /* Phase one deliberately supports only one unique virtual value in the
     * current frame's operand stack.  A value in a local or suspended caller,
     * an alias, or multiple raw values needs exit reconstruction and therefore
     * uses the non-virtual retry instead. */
    int stack_virtuals = _Py_uop_alloc_context_stack_virtual_count(
        ctx, stack_pointer);
    int all_virtuals = _Py_uop_alloc_context_virtual_count(
        ctx, stack_pointer);
    if (stack_virtuals != 1 || all_virtuals != stack_virtuals) {
        return false;
    }

    JitOptRef *slot = NULL;
    for (JitOptRef *ref = ctx->frame->stack; ref < stack_pointer; ref++) {
        if (_Py_uop_alloc_is_virtual(ctx, *ref)) {
            slot = ref;
            break;
        }
    }
    if (slot == NULL || !PyJitRef_IsUnique(*slot)) {
        return false;
    }

    /* _MATERIALIZE_VIRTUAL_FLOAT consumes TOS.  Tier 2 provides _SWAP forms
     * for depths two through four, so temporarily move a deeper raw value to
     * TOS and put the resulting box back in the same logical stack slot. */
    int depth = (int)(stack_pointer - slot);
    if (depth < 1 || depth > 4) {
        return false;
    }
    int required_output_slots = depth == 1 ? 2 : 4;
    if (ctx->out_buffer.end - ctx->out_buffer.next < required_output_slots) {
        return false;
    }
    JitOptRef concrete = _Py_uop_alloc_materialize_virtual_float(ctx, *slot);
    if (PyJitRef_IsNull(concrete) || PyJitRef_IsInvalid(concrete)) {
        return false;
    }

    JitOptRef *top = stack_pointer - 1;
    if (depth > 1) {
        add_op(ctx, this_instr, _SWAP, (uint16_t)depth, 0);
        JitOptRef temporary = *slot;
        *slot = *top;
        *top = temporary;
    }
    add_op(ctx, this_instr, _MATERIALIZE_VIRTUAL_FLOAT, 0, 0);
    *top = concrete;
    if (depth > 1) {
        add_op(ctx, this_instr, _SWAP, (uint16_t)depth, 0);
        JitOptRef temporary = *slot;
        *slot = *top;
        *top = temporary;
    }
    return true;
}

/* Shortened forms for convenience, used in optimizer_bytecodes.c */
#define sym_is_not_null _Py_uop_sym_is_not_null
#define sym_is_const _Py_uop_sym_is_const
#define sym_is_safe_const _Py_uop_sym_is_safe_const
#define sym_is_not_container _Py_uop_sym_is_not_container
#define sym_get_const _Py_uop_sym_get_const
#define sym_new_const_steal _Py_uop_sym_new_const_steal
#define sym_get_const_as_stackref _Py_uop_sym_get_const_as_stackref
#define sym_new_unknown _Py_uop_sym_new_unknown
#define sym_new_not_null _Py_uop_sym_new_not_null
#define sym_new_type _Py_uop_sym_new_type
#define sym_is_null _Py_uop_sym_is_null
#define sym_new_const _Py_uop_sym_new_const
#define sym_new_null _Py_uop_sym_new_null
#define sym_has_type _Py_uop_sym_has_type
#define sym_get_type _Py_uop_sym_get_type
#define sym_get_probable_type _Py_uop_sym_get_probable_type
#define sym_matches_type _Py_uop_sym_matches_type
#define sym_matches_type_version _Py_uop_sym_matches_type_version
#define sym_get_type_version _Py_uop_sym_get_type_version
#define sym_set_null(SYM) _Py_uop_sym_set_null(ctx, SYM)
#define sym_set_non_null(SYM) _Py_uop_sym_set_non_null(ctx, SYM)
#define sym_set_type(SYM, TYPE) _Py_uop_sym_set_type(ctx, SYM, TYPE)
#define sym_set_type_version(SYM, VERSION) _Py_uop_sym_set_type_version(ctx, SYM, VERSION)
#define sym_set_const(SYM, CNST) _Py_uop_sym_set_const(ctx, SYM, CNST)
#define sym_set_compact_int(SYM) _Py_uop_sym_set_compact_int(ctx, SYM)
#define sym_is_bottom _Py_uop_sym_is_bottom
#define sym_truthiness _Py_uop_sym_truthiness
#define frame_new _Py_uop_frame_new
#define frame_new_from_symbol _Py_uop_frame_new_from_symbol
#define frame_pop _Py_uop_frame_pop
#define sym_new_tuple _Py_uop_sym_new_tuple
#define sym_tuple_getitem _Py_uop_sym_tuple_getitem
#define sym_tuple_length _Py_uop_sym_tuple_length
#define sym_is_immortal _Py_uop_symbol_is_immortal
#define sym_is_compact_int _Py_uop_sym_is_compact_int
#define sym_new_compact_int _Py_uop_sym_new_compact_int
#define sym_new_truthiness _Py_uop_sym_new_truthiness
#define sym_new_predicate _Py_uop_sym_new_predicate
#define sym_apply_predicate_narrowing _Py_uop_sym_apply_predicate_narrowing
#define sym_set_recorded_type(SYM, TYPE) _Py_uop_sym_set_recorded_type(ctx, SYM, TYPE)
#define sym_set_recorded_value(SYM, VAL) _Py_uop_sym_set_recorded_value(ctx, SYM, VAL)
#define sym_set_recorded_gen_func(SYM, VAL) _Py_uop_sym_set_recorded_gen_func(ctx, SYM, VAL)
#define sym_get_probable_func_code _Py_uop_sym_get_probable_func_code
#define sym_get_probable_value _Py_uop_sym_get_probable_value
#define sym_set_stack_depth(DEPTH, SP) _Py_uop_sym_set_stack_depth(ctx, DEPTH, SP)

/* Comparison oparg masks */
#define COMPARE_LT_MASK 2
#define COMPARE_GT_MASK 4
#define COMPARE_EQ_MASK 8

#define JUMP_TO_LABEL(label) goto label;

static int
check_stack_bounds(JitOptContext *ctx, JitOptRef *stack_pointer, int offset, int opcode)
{
    int stack_level = (int)(stack_pointer + (offset) - ctx->frame->stack);
    int should_check = !CURRENT_FRAME_IS_INIT_SHIM() ||
        (opcode == _RETURN_VALUE) ||
        (opcode == _RETURN_GENERATOR) ||
        (opcode == _YIELD_VALUE);
    if (should_check && (stack_level < 0 || stack_level > STACK_SIZE() + MAX_CACHED_REGISTER)) {
        ctx->contradiction = true;
        ctx->done = true;
        return 1;
    }
    return 0;
}

#define CHECK_STACK_BOUNDS(offset) \
    if (check_stack_bounds(ctx, stack_pointer, offset, opcode)) { \
        break; \
    } \

static int
optimize_to_bool(
    _PyUOpInstruction *this_instr,
    JitOptContext *ctx,
    JitOptRef value,
    JitOptRef *result_ptr,
    uint16_t prefix, uint16_t suffix)
{
    if (sym_matches_type(value, &PyBool_Type)) {
        ADD_OP(_NOP, 0, 0);
        *result_ptr = value;
        return 1;
    }
    int truthiness = sym_truthiness(ctx, value);
    if (truthiness >= 0) {
        PyObject *load = truthiness ? Py_True : Py_False;
        if (prefix != _NOP) {
            ADD_OP(prefix, 0, 0);
        }
        ADD_OP(_LOAD_CONST_INLINE_BORROW, 0, (uintptr_t)load);
        if (suffix != _NOP) {
            ADD_OP(suffix, 2, 0);
        }
        *result_ptr = sym_new_const(ctx, load);
        return 1;
    }
    return 0;
}

static void
optimize_dict_known_hash(
    JitOptContext *ctx, _PyBloomFilter *dependencies, _PyUOpInstruction *this_instr,
    PyObject *sub, uint16_t opcode)
{
    if (PyUnicode_CheckExact(sub) || PyLong_CheckExact(sub) || PyBytes_CheckExact(sub)
            || PyFloat_CheckExact(sub) || PyComplex_CheckExact(sub)) {
        // PyObject_Hash can't fail on these types
        ADD_OP(opcode, 0, PyObject_Hash(sub));
    }
    else if (PyTuple_CheckExact(sub)) {
        // only use known hash variant when hash of tuple is already computed
        // since computing it can call arbitrary code
        Py_hash_t hash = ((PyTupleObject *)sub)->ob_hash;
        if (hash != -1) {
            ADD_OP(opcode, 0, hash);
        }
    }
    else if (Py_TYPE(sub)->tp_hash == PyBaseObject_Type.tp_hash) {
        // for user-defined objects which don't override tp_hash
        Py_hash_t hash = PyObject_Hash(sub);
        ADD_OP(opcode, 0, hash);
        watch_type(Py_TYPE(sub), dependencies);
    }
}

static void
eliminate_pop_guard(_PyUOpInstruction *this_instr, JitOptContext *ctx, bool exit)
{
    ADD_OP(_POP_TOP, 0, 0);
    if (exit) {
        REPLACE_OP((this_instr+1), _EXIT_TRACE, 0, 0);
        this_instr[1].target = this_instr->target;
    }
}

static JitOptRef
lookup_attr(JitOptContext *ctx, _PyBloomFilter *dependencies, _PyUOpInstruction *this_instr,
            PyTypeObject *type, PyObject *name,
            uint16_t prefix, uint16_t suffix)
{
    // The cached value may be dead, so we need to do the lookup again... :(
    if (type && PyType_Check(type)) {
        PyObject *lookup = _PyType_Lookup(type, name);
        if (lookup) {
            bool immortal = _Py_IsImmortal(lookup) || (type->tp_flags & Py_TPFLAGS_IMMUTABLETYPE);
            if (prefix != _NOP) {
                ADD_OP(prefix, 0, 0);
            }
            ADD_OP(immortal ? _LOAD_CONST_INLINE_BORROW : _LOAD_CONST_INLINE,
                   0, (uintptr_t)lookup);
            if (suffix != _NOP) {
                ADD_OP(suffix, 2, 0);
            }
            if ((type->tp_flags & Py_TPFLAGS_IMMUTABLETYPE) == 0) {
                watch_type(type, dependencies);
            }
            return sym_new_const(ctx, lookup);
        }
    }
    return sym_new_not_null(ctx);
}

static void
optimize_pop_top(JitOptContext *ctx, _PyUOpInstruction *this_instr, JitOptRef value)
{
    if (_Py_uop_alloc_is_virtual(ctx, value)) {
        const JitVirtualObject *virtual_object =
            _Py_uop_alloc_resolve_virtual(ctx, value);
        assert(virtual_object != NULL);
        assert(virtual_object->kind == JIT_VIRTUAL_FLOAT);
        assert(sym_get_type(value) == &PyFloat_Type);
        ADD_OP(_POP_TOP_VIRTUAL_FLOAT, 0, 0);
        return;
    }
    PyTypeObject *typ = sym_get_type(value);
    if (PyJitRef_IsBorrowed(value) ||
        sym_is_immortal(PyJitRef_Unwrap(value)) ||
        sym_is_null(value)) {
        ADD_OP(_POP_TOP_NOP, 0, 0);
    }
    else if (typ == &PyLong_Type) {
        ADD_OP(_POP_TOP_INT, 0, 0);
    }
    else if (typ == &PyFloat_Type) {
        ADD_OP(_POP_TOP_FLOAT, 0, 0);
    }
    else if (typ == &PyUnicode_Type) {
        ADD_OP(_POP_TOP_UNICODE, 0, 0);
    }
    else {
        ADD_OP(_POP_TOP, 0, 0);
    }
}

bool
_Py_uop_can_virtualize_float_result(JitOptContext *ctx)
{
    if (!ctx->virtual_floats_enabled) {
        return false;
    }
    /* A NULL plan is the legacy optimistic policy.  The profiling pass also
     * admits every candidate so it can observe the complete raw-value graph. */
    if (ctx->virtualization_plan == NULL ||
        ctx->collect_virtualization_profitability)
    {
        return true;
    }
    /* Pass two makes the decision at the same input-trace position.  Missing
     * entries are rejected conservatively: they were not proven profitable
     * by the optimistic pass. */
    int producer = ctx->current_trace_index;
    return producer >= 0 && producer < UOP_MAX_TRACE_LENGTH &&
        ctx->virtualization_plan->producer_seen[producer] &&
        ctx->virtualization_plan->producer_profitable[producer];
}

/* Look up name via super (normal case from supercheck where
   su_obj_type = Py_TYPE(obj)). */
static JitOptRef
lookup_super_attr(JitOptContext *ctx, _PyBloomFilter *dependencies,
                  _PyUOpInstruction *this_instr,
                  PyTypeObject *su_type, PyTypeObject *obj_type,
                  PyObject *name,
                  uint16_t immortal, uint16_t mortal, uint16_t suffix)
{
    if (su_type == NULL || obj_type == NULL) {
        return sym_new_not_null(ctx);
    }
    /* Normal case: obj_type must be a subtype of su_type */
    if (!PyType_IsSubtype(obj_type, su_type)) {
        return sym_new_not_null(ctx);
    }
    PyObject *lookup = _PySuper_LookupDescr(su_type, obj_type, name);
    if (lookup == NULL) {
        if (PyErr_Occurred()) {
            PyErr_Clear();
        }
        return sym_new_not_null(ctx);
    }
    if ((Py_TYPE(lookup)->tp_flags & Py_TPFLAGS_METHOD_DESCRIPTOR) == 0) {
        Py_DECREF(lookup);
        return sym_new_not_null(ctx);
    }
    int opcode = mortal;
    if (_Py_IsImmortal(lookup) || (obj_type->tp_flags & Py_TPFLAGS_IMMUTABLETYPE)) {
        opcode = immortal;
    }
    ADD_OP(_SWAP, 3, 0);
    ADD_OP(_POP_TOP, 0, 0);
    ADD_OP(_POP_TOP, 0, 0);
    ADD_OP(opcode, 0, (uintptr_t)lookup);
    if (suffix != _NOP) {
        ADD_OP(suffix, 2, 0);
    }
    // if obj_type is immutable, then all its superclasses are immutable
    if ((obj_type->tp_flags & Py_TPFLAGS_IMMUTABLETYPE) == 0) {
        watch_type(su_type, dependencies);
        watch_type(obj_type, dependencies);
    }
    return sym_new_const_steal(ctx, lookup);
}

static
PyCodeObject *
get_current_code_object(JitOptContext *ctx)
{
    return (PyCodeObject *)ctx->frame->code;
}

static PyObject *
get_co_name(JitOptContext *ctx, int index)
{
    return PyTuple_GET_ITEM(get_current_code_object(ctx)->co_names, index);
}

static int
get_test_bit_for_bools(void) {
#ifdef Py_STACKREF_DEBUG
    uintptr_t false_bits = _Py_STACKREF_FALSE_INDEX;
    uintptr_t true_bits = _Py_STACKREF_TRUE_INDEX;
#else
    uintptr_t false_bits = (uintptr_t)&_Py_FalseStruct;
    uintptr_t true_bits = (uintptr_t)&_Py_TrueStruct;
#endif
    for (int i = 4; i < 8; i++) {
        if ((true_bits ^ false_bits) & (uintptr_t)(1 << i)) {
            return i;
        }
    }
    return 0;
}

static int
test_bit_set_in_true(int bit) {
#ifdef Py_STACKREF_DEBUG
    uintptr_t true_bits = _Py_STACKREF_TRUE_INDEX;
#else
    uintptr_t true_bits = (uintptr_t)&_Py_TrueStruct;
#endif
    assert((true_bits ^ ((uintptr_t)&_Py_FalseStruct)) & (uintptr_t)(1 << bit));
    return true_bits & (uintptr_t)(1 << bit);
}

#ifdef Py_DEBUG
void
_Py_opt_assert_within_stack_bounds(
    _Py_UOpsAbstractFrame *frame, JitOptRef *stack_pointer,
    const char *filename, int lineno
) {
    if (frame->code == ((PyCodeObject *)&_Py_InitCleanup)) {
        return;
    }
    int level = (int)(stack_pointer - frame->stack);
    if (level < 0) {
        printf("Stack underflow (depth = %d) at %s:%d\n", level, filename, lineno);
        fflush(stdout);
        abort();
    }
    int size = (int)(frame->stack_len) + MAX_CACHED_REGISTER;
    if (level > size) {
        printf("Stack overflow (depth = %d) at %s:%d\n", level, filename, lineno);
        fflush(stdout);
        abort();
    }
}
#endif

#ifdef Py_DEBUG
#define ASSERT_WITHIN_STACK_BOUNDS(F, L) _Py_opt_assert_within_stack_bounds(ctx->frame, stack_pointer, (F), (L))
#else
#define ASSERT_WITHIN_STACK_BOUNDS(F, L) (void)0
#endif

/* >0 (length) for success, 0 for not ready, clears all possible errors. */
static int
optimize_uops(
    _PyThreadStateImpl *tstate,
    _PyUOpInstruction *trace,
    int trace_len,
    int curr_stacklen,
    _PyUOpInstruction *output,
    _PyBloomFilter *dependencies,
    bool enable_virtual_floats,
    JitVirtualizationPlan *virtualization_plan,
    bool collect_virtualization_profitability
)
{
    assert(!PyErr_Occurred());
    assert(tstate->jit_tracer_state != NULL);
    PyFunctionObject *func = tstate->jit_tracer_state->initial_state.func;

    JitOptContext *ctx = &tstate->jit_tracer_state->opt_context;
    uint32_t opcode = UINT16_MAX;

    uop_buffer_init(&ctx->out_buffer, output, UOP_MAX_TRACE_LENGTH);

    // Make sure that watchers are set up
    PyInterpreterState *interp = _PyInterpreterState_GET();
    _PyOnceFlag_CallOnce(&interp->dict_state.watcher_setup_once,
                         _setup_optimizer_watchers, NULL);

    _Py_uop_abstractcontext_init(ctx, dependencies);
    ctx->virtualization_plan = virtualization_plan;
    ctx->collect_virtualization_profitability =
        collect_virtualization_profitability;
    /* Platform support and debug-stack compatibility are checked once per
     * pass; individual producer rules can then use a cheap boolean gate. */
    ctx->virtual_floats_enabled =
        enable_virtual_floats && _Py_JIT_RAW_DOUBLE_SUPPORTED;
#ifdef Py_DEBUG
    /* The Tier 2 interpreter's high-level trace mode treats every live slot
     * as a Python object.  Do not put raw double bits in such an executor. */
    if (tstate->base.current_frame != NULL &&
        tstate->base.current_frame->lltrace >= 4)
    {
        ctx->virtual_floats_enabled = false;
    }
#endif
    _Py_UOpsAbstractFrame *frame = _Py_uop_frame_new(ctx, (PyCodeObject *)func->func_code, NULL, 0);
    if (frame == NULL) {
        return 0;
    }
    frame->func = func;
    ctx->curr_frame_depth++;
    ctx->frame = frame;
    _Py_uop_sym_set_stack_depth(ctx, curr_stacklen, frame->stack_pointer);

    _PyUOpInstruction *this_instr = NULL;
    JitOptRef *stack_pointer = ctx->frame->stack_pointer;
    bool reached_loop_backedge = false;

    for (int i = 0; i < trace_len; i++) {
        this_instr = &trace[i];
        ctx->current_trace_index = i;
        if (ctx->done) {
            // Don't do any more optimization, but
            // we still need to reach a terminator for corrctness.
            *(ctx->out_buffer.next++) = *this_instr;
            if (is_terminator_uop(this_instr)) {
                break;
            }
            continue;
        }

        int oparg = this_instr->oparg;
        opcode = this_instr->opcode;

        if (!CURRENT_FRAME_IS_INIT_SHIM()) {
            stack_pointer = ctx->frame->stack_pointer;
        }

        DUMP_UOP(ctx, "abs", (int)(this_instr - trace), this_instr, stack_pointer);

        _PyUOpInstruction *out_ptr = ctx->out_buffer.next;
        /* Snapshot both stack and local liveness before rewriting this uop.
         * The counts drive pre-barrier materialization and post-rule audits. */
        int live_virtuals_before = _Py_uop_alloc_context_stack_virtual_count(
            ctx, stack_pointer);

        int all_virtuals_before = _Py_uop_alloc_context_virtual_count(
            ctx, stack_pointer);
        int local_virtuals_before = all_virtuals_before < 0 ||
            live_virtuals_before < 0
            ? -1 : all_virtuals_before - live_virtuals_before;

        /* Calls and other escaping operations may inspect the Python frame.
         * Restore shadow locals before those operations execute.  If a raw
         * stack value aliases a shadow local, this first implementation cannot
         * rebuild their shared identity, so retry the trace without virtuals. */
        if (ctx->virtual_floats_enabled && local_virtuals_before != 0 &&
            (_PyUop_Flags[this_instr->opcode] & HAS_ESCAPES_FLAG) != 0 &&
            !uop_handles_virtual_float(this_instr) &&
            !uop_preserves_virtual_fast_locals(this_instr))
        {
            if (local_virtuals_before < 0 || live_virtuals_before != 0 ||
                ctx->curr_frame_depth != 1)
            {
                ctx->retry_without_virtuals = true;
                ctx->done = true;
                break;
            }
            for (int local = 0; local < ctx->frame->locals_len &&
                                local < _PY_JIT_MAX_VIRTUAL_FAST_LOCALS;
                 local++)
            {
                if (_Py_uop_alloc_is_virtual(ctx, ctx->frame->locals[local])) {
                    _Py_uop_alloc_note_materialization(
                        ctx, ctx->frame->locals[local]);
                    add_op(ctx, this_instr,
                           _MATERIALIZE_VIRTUAL_FAST_LOCAL, local, 0);
                    ctx->frame->locals[local] =
                        sym_new_type(ctx, &PyFloat_Type);
                }
            }
        }

        /* A generic barrier may proceed only after its sole unique stack
         * virtual has been boxed.  More complex states use the safe retry. */
        if (ctx->virtual_floats_enabled && live_virtuals_before != 0 &&
            is_virtualization_barrier(this_instr) &&
            !uop_handles_virtual_float(this_instr) &&
            !materialize_single_stack_virtual(
                ctx, this_instr, stack_pointer))
        {
            DPRINTF(1,
                "virtual float could not be materialized before escape; "
                "retrying\n");
            ctx->retry_without_virtuals = true;
            ctx->done = true;
            break;
        }

        /* Prefix materialization does not replace the source uop.  Remember
         * where the generated transfer rule starts so a rule which emits
         * nothing still copies the original consumer after the prefix. */
        _PyUOpInstruction *source_out_ptr = ctx->out_buffer.next;
        switch (opcode) {

#include "optimizer_cases.c.h"

            default:
                DPRINTF(1, "\nUnknown opcode in abstract interpreter\n");
                Py_UNREACHABLE();
        }
        if (opcode == _JUMP_TO_TOP) {
            reached_loop_backedge = true;
        }
        // If no ADD_OP was called during this iteration, copy the original instruction
        if (ctx->out_buffer.next == source_out_ptr) {
            *(ctx->out_buffer.next++) = *this_instr;
        }
        assert(ctx->frame != NULL);
        bool live_virtuals_after =
            _Py_uop_alloc_context_stack_virtual_count(
                ctx, stack_pointer) != 0;
        if (ctx->virtual_floats_enabled &&
            virtuals_cross_unsafe_barrier(
                out_ptr, ctx->out_buffer.next,
                live_virtuals_before, live_virtuals_after))
        {
            DPRINTF(1, "virtual float reached an unsafe barrier; retrying\n");
            ctx->retry_without_virtuals = true;
            ctx->done = true;
            break;
        }
        DUMP_UOPS(ctx, "out", out_ptr, stack_pointer);
        if (!CURRENT_FRAME_IS_INIT_SHIM() && !ctx->done) {
            assert(_Py_uop_alloc_validate_context(ctx, stack_pointer));
            DPRINTF(3, " stack_level %d\n", STACK_LEVEL());
            ctx->frame->stack_pointer = stack_pointer;
            assert(STACK_LEVEL() >= 0);
        }
    }
    if (ctx->retry_without_virtuals) {
        _Py_uop_abstractcontext_fini(ctx);
        return OPTIMIZE_RETRY_WITHOUT_VIRTUALS;
    }
    if (ctx->out_of_space) {
        DPRINTF(3, "\n");
        DPRINTF(1, "Out of space in abstract interpreter\n");
    }
    if (ctx->contradiction) {
        // Attempted to push a "bottom" (contradiction) symbol onto the stack.
        // This means that the abstract interpreter has optimized to trace
        // to an unreachable estate.
        // We *could* generate an _EXIT_TRACE or _FATAL_ERROR here, but hitting
        // bottom usually indicates an optimizer bug, so we are probably better off
        // retrying later.
        DPRINTF(3, "\n");
        DPRINTF(1, "Hit bottom in abstract interpreter\n");
        _Py_uop_abstractcontext_fini(ctx);
        OPT_STAT_INC(optimizer_contradiction);
        return 0;
    }

    /* A loop executor may retain write-before-read root-frame locals in the
     * runtime shadow table across its direct backedge.  Operand-stack values,
     * inlined-frame values, unsafe locals, and every non-loop terminator still
     * require ordinary boxed exit reconstruction. */
    bool safe_loop_shadows = false;
    if (reached_loop_backedge &&
        ctx->curr_frame_depth == 1 &&
        _Py_uop_alloc_context_stack_virtual_count(ctx, stack_pointer) == 0)
    {
        safe_loop_shadows = true;
        for (int local = 0; local < ctx->frame->locals_len; local++) {
            if (_Py_uop_alloc_is_virtual(ctx, ctx->frame->locals[local]) &&
                (local >= _PY_JIT_MAX_VIRTUAL_FAST_LOCALS ||
                 !ctx->virtual_fast_local_loop_safe[local]))
            {
                safe_loop_shadows = false;
                break;
            }
        }
    }
    if (_Py_uop_alloc_context_has_virtual(ctx, stack_pointer) &&
        !safe_loop_shadows)
    {
        DPRINTF(1, "virtual float remained live at the end; retrying\n");
        _Py_uop_abstractcontext_fini(ctx);
        return OPTIMIZE_RETRY_WITHOUT_VIRTUALS;
    }

    if (ctx->collect_virtualization_profitability &&
        ctx->virtualization_plan != NULL)
    {
        /* Export decisions before fini discards the trace-local descriptors. */
        _Py_uop_alloc_build_profitability_plan(ctx);
    }

    /* Either reached the end or cannot optimize further, but there
     * would be no benefit in retrying later */
    _Py_uop_abstractcontext_fini(ctx);
    // Check that the trace ends with a proper terminator
    if (uop_buffer_length(&ctx->out_buffer) > 0) {
        assert(is_terminator_uop(uop_buffer_last(&ctx->out_buffer)));
    }

    return uop_buffer_length(&ctx->out_buffer);

error:
    DPRINTF(3, "\n");
    DPRINTF(1, "Encountered error in abstract interpreter\n");
    if (opcode <= MAX_UOP_ID) {
        OPT_ERROR_IN_OPCODE(opcode);
    }
    _Py_uop_abstractcontext_fini(ctx);

    assert(PyErr_Occurred());
    PyErr_Clear();

    return 0;

}

const uint16_t op_without_push[MAX_UOP_ID + 1] = {
    [_COPY] = _NOP,
    [_LOAD_CONST_INLINE] = _NOP,
    [_LOAD_CONST_INLINE_BORROW] = _NOP,
    [_LOAD_FAST] = _NOP,
    [_LOAD_FAST_BORROW] = _NOP,
    [_LOAD_SMALL_INT] = _NOP,
    [_PUSH_NULL] = _NOP,
};

const bool op_skip[MAX_UOP_ID + 1] = {
    [_NOP] = true,
    [_CHECK_VALIDITY] = true,
    [_CHECK_PERIODIC] = true,
    [_SET_IP] = true,
};

const uint16_t op_without_pop[MAX_UOP_ID + 1] = {
    [_POP_TOP] = _NOP,
    [_POP_TOP_NOP] = _NOP,
    [_POP_TOP_INT] = _NOP,
    [_POP_TOP_FLOAT] = _NOP,
    [_POP_TOP_UNICODE] = _NOP,
};


static int
remove_unneeded_uops(_PyUOpInstruction *buffer, int buffer_size)
{
    /* Remove _SET_IP and _CHECK_VALIDITY where possible.
     * _SET_IP is needed if the following instruction escapes or
     * could error. _CHECK_VALIDITY is needed if the previous
     * instruction could have escaped. */
    int last_set_ip = -1;
    bool may_have_escaped = true;
    for (int pc = 0; pc < buffer_size; pc++) {
        int opcode = buffer[pc].opcode;
        switch (opcode) {
            case _START_EXECUTOR:
                may_have_escaped = false;
                break;
            case _SET_IP:
                buffer[pc].opcode = _NOP;
                last_set_ip = pc;
                break;
            case _CHECK_VALIDITY:
                if (may_have_escaped) {
                    may_have_escaped = false;
                }
                else {
                    buffer[pc].opcode = _NOP;
                }
                break;
            case _EXIT_TRACE:
            default:
            {
                // Cancel out pushes and pops, repeatedly. So:
                //     _LOAD_FAST + _POP_TOP + _POP_TOP + _LOAD_CONST_INLINE_BORROW + _POP_TOP
                // ...becomes:
                //     _NOP + _NOP + _POP_TOP + _NOP + _NOP
                while (op_without_pop[opcode]) {
                    _PyUOpInstruction *last = &buffer[pc - 1];
                    while (op_skip[last->opcode]) {
                        last--;
                    }
                    if (op_without_push[last->opcode] && op_without_pop[opcode]) {
                        last->opcode = op_without_push[last->opcode];
                        opcode = buffer[pc].opcode = op_without_pop[opcode];
                        if (op_without_pop[last->opcode]) {
                            opcode = last->opcode;
                            pc = (int)(last - buffer);
                        }
                    }
                    else {
                        break;
                    }
                }
                /* _PUSH_FRAME doesn't escape or error, but it
                 * does need the IP for the return address */
                bool needs_ip = (opcode == _PUSH_FRAME || opcode == _YIELD_VALUE || opcode == _DYNAMIC_EXIT || opcode == _EXIT_TRACE);
                if (_PyUop_Flags[opcode] & HAS_ESCAPES_FLAG) {
                    needs_ip = true;
                    may_have_escaped = true;
                }
                if (needs_ip && last_set_ip >= 0) {
                    assert(buffer[last_set_ip].opcode == _NOP);
                    buffer[last_set_ip].opcode = _SET_IP;
                    last_set_ip = -1;
                }
                if (opcode == _EXIT_TRACE) {
                    return pc + 1;
                }
                break;
            }
            case _JUMP_TO_TOP:
            case _DYNAMIC_EXIT:
            case _DEOPT:
                return pc + 1;
        }
    }
    Py_UNREACHABLE();
}

//  0 - failure, no error raised, just fall back to Tier 1
// -1 - failure, and raise error
//  > 0 - length of optimized trace
int
_Py_uop_analyze_and_optimize(
    _PyThreadStateImpl *tstate,
    _PyUOpInstruction *buffer,
    int length,
    int curr_stacklen,
    _PyUOpInstruction *output,
    _PyBloomFilter *dependencies
)
{
    OPT_STAT_INC(optimizer_attempts);

    enum {
        VIRTUAL_FLOATS_DISABLED,
        VIRTUAL_FLOATS_HEURISTIC,
        VIRTUAL_FLOATS_OPTIMISTIC,
    } virtual_float_mode = VIRTUAL_FLOATS_HEURISTIC;
    /* Private A/B switch for the virtual-float experiment.  Reading it while
     * compiling a trace does not affect the generated executor's hot path.
     * 0 disables virtualization, 2/optimistic selects the old policy, and
     * unset/1/heuristic selects the profitability pass. */
    char *virtual_floats = Py_GETENV("PYTHON_JIT_VIRTUAL_FLOATS");
    if (virtual_floats != NULL && *virtual_floats == '0') {
        virtual_float_mode = VIRTUAL_FLOATS_DISABLED;
    }
    else if (virtual_floats != NULL &&
             (*virtual_floats == '2' ||
              strcmp(virtual_floats, "optimistic") == 0))
    {
        virtual_float_mode = VIRTUAL_FLOATS_OPTIMISTIC;
    }

    int trace_length = length;
    if (virtual_float_mode == VIRTUAL_FLOATS_HEURISTIC) {
        /* The first pass measures a complete optimistic virtual graph.  The
         * second pass starts from fresh abstract state and emits only the
         * connected components whose virtual work exceeds reconstruction. */
        JitVirtualizationPlan plan = {0};
        length = optimize_uops(
            tstate, buffer, trace_length, curr_stacklen, output, dependencies,
            true, &plan, true);
        bool has_virtual_candidates = false;
        bool has_profitable_candidates = false;
        for (int i = 0; i < trace_length; i++) {
            if (plan.producer_seen[i]) {
                has_virtual_candidates = true;
                if (plan.producer_profitable[i]) {
                    has_profitable_candidates = true;
                }
            }
        }
        /* Avoid a redundant optimizer pass for the overwhelmingly common
         * traces with no virtual-float producer.  Besides reducing compile
         * cost, this preserves the optimizer's established single-pass
         * handling of mutable profiling facts and dependency watchers. */
        if (length > 0 && has_profitable_candidates) {
            length = optimize_uops(
                tstate, buffer, trace_length, curr_stacklen, output,
                dependencies, true, &plan, false);
        }
        else if (length > 0 && has_virtual_candidates) {
            /* If every component lost the profitability test, generate the
             * ordinary boxed trace directly.  This makes reject-all exactly
             * equivalent to mode 0 instead of replaying a nominally virtual
             * pass which happens not to select any producers. */
            length = optimize_uops(
                tstate, buffer, trace_length, curr_stacklen, output,
                dependencies, false, NULL, false);
        }
    }
    else {
        length = optimize_uops(
            tstate, buffer, trace_length, curr_stacklen, output, dependencies,
            virtual_float_mode == VIRTUAL_FLOATS_OPTIMISTIC, NULL, false);
    }

    if (length == OPTIMIZE_RETRY_WITHOUT_VIRTUALS) {
        /* The first pass is speculative.  Reinitialize every abstract arena
         * and regenerate the trace using CPython's ordinary boxed uops. */
        assert(virtual_float_mode != VIRTUAL_FLOATS_DISABLED);
        length = optimize_uops(
            tstate, buffer, trace_length, curr_stacklen, output, dependencies,
            false, NULL, false);
    }

    if (length == 0) {
        return length;
    }

    assert(length > 0);

    length = remove_unneeded_uops(output, length);
    assert(length > 0);

    OPT_STAT_INC(optimizer_successes);
    return length;
}

#endif /* _Py_TIER2 */
