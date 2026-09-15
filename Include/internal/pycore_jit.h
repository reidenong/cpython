#ifndef Py_INTERNAL_JIT_H
#define Py_INTERNAL_JIT_H

#ifdef __cplusplus
extern "C" {
#endif

#include "pycore_interp.h"
#include "pycore_optimizer.h"
#include "pycore_stackref.h"

#include <float.h>   // DBL_MANT_DIG
#include <string.h>  // memcpy()

#ifndef Py_BUILD_CORE
#  error "this header requires Py_BUILD_CORE define"
#endif

/* A virtual float's double payload is transported between dedicated Tier 2
 * uops in the bits of an otherwise invalid _PyStackRef.  Code handling this
 * value must treat it as opaque: only these helpers and representation-aware
 * uops may inspect it.  In particular, it must never reach PyStackRef_* APIs,
 * a side exit, frame synchronization, or the generic stack debugger.
 *
 * Py_STACKREF_DEBUG uses the bits as an index into its reference table, so it
 * is deliberately excluded.  Unsupported builds can still compile Tier 2,
 * but the optimizer must not emit a raw-double producer on them.
 */
#if SIZEOF_VOID_P == 8 && SIZEOF_DOUBLE == 8 \
    && FLT_RADIX == 2 && DBL_MANT_DIG == 53 && DBL_MAX_EXP == 1024 \
    && !defined(Py_STACKREF_DEBUG)
#  define _Py_JIT_RAW_DOUBLE_SUPPORTED 1

static inline _PyStackRef
_PyJIT_PackRawDouble(double value)
{
    static_assert(sizeof(_PyStackRef) == sizeof(value),
                  "raw float transport requires a 64-bit stack slot");
    _PyStackRef result;
    memcpy(&result, &value, sizeof(value));
    return result;
}

static inline double
_PyJIT_UnpackRawDouble(_PyStackRef value)
{
    double result;
    memcpy(&result, &value, sizeof(result));
    return result;
}
#else
#  define _Py_JIT_RAW_DOUBLE_SUPPORTED 0

/* Keep the Tier 2 uop definitions buildable on unsupported targets.  The
 * optimizer feature gate guarantees that these stubs are unreachable. */
static inline _PyStackRef
_PyJIT_PackRawDouble(double value)
{
    (void)value;
    Py_UNREACHABLE();
}

static inline double
_PyJIT_UnpackRawDouble(_PyStackRef value)
{
    (void)value;
    Py_UNREACHABLE();
}
#endif

/* Representation mask carried by virtual-float arithmetic uops. */
#define _Py_JIT_VIRTUAL_FLOAT_RAW_LEFT 1
#define _Py_JIT_VIRTUAL_FLOAT_RAW_RIGHT 2

/* COMPARE_OP already uses bits 0 through 7 for its comparison operation,
 * result mask, and boolean-conversion flag.  Its virtual-float representation
 * mask therefore lives in the next two bits. */
#define _Py_JIT_COMPARE_VIRTUAL_FLOAT_RAW_LEFT (1 << 8)
#define _Py_JIT_COMPARE_VIRTUAL_FLOAT_RAW_RIGHT (1 << 9)
#define _Py_JIT_COMPARE_VIRTUAL_FLOAT_RAW_MASK \
    (_Py_JIT_COMPARE_VIRTUAL_FLOAT_RAW_LEFT | \
     _Py_JIT_COMPARE_VIRTUAL_FLOAT_RAW_RIGHT)

/* To be able to reason about code layout and branches, keep code size below 1 MB */
#define PY_MAX_JIT_CODE_SIZE ((1 << 20)-1)

#ifdef _Py_JIT

typedef _Py_CODEUNIT *(*jit_func)(
    _PyExecutorObject *executor, _PyInterpreterFrame *frame, _PyStackRef *stack_pointer, PyThreadState *tstate,
    _PyStackRef _tos_cache0, _PyStackRef _tos_cache1, _PyStackRef _tos_cache2
);

_Py_CODEUNIT *_PyJIT_Entry(
    _PyExecutorObject *executor, _PyInterpreterFrame *frame,
    _PyStackRef *stack_pointer, PyThreadState *tstate
);

int _PyJIT_Compile(_PyExecutorObject *executor, const _PyUOpInstruction *trace, size_t length);
void _PyJIT_Free(_PyExecutorObject *executor);
PyAPI_FUNC(int) _PyJIT_AddressInJitCode(PyInterpreterState *interp, uintptr_t addr);
PyAPI_FUNC(void) _Py_jit_assert_within_stack_bounds(_PyInterpreterFrame *frame, _PyStackRef *stack_pointer, int lineno);
PyAPI_FUNC(int) _Py_jit_assertion_failure(int line);

#endif  // _Py_JIT

#ifdef __cplusplus
}
#endif

#endif // !Py_INTERNAL_JIT_H
