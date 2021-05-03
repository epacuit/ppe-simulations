#include "init.c"
#include "getargs.c"
#include "int_ops.c"
#include "list_ops.c"
#include "dict_ops.c"
#include "str_ops.c"
#include "set_ops.c"
#include "tuple_ops.c"
#include "exc_ops.c"
#include "misc_ops.c"
#include "generic_ops.c"
#include "__native.h"
#include "__native_internal.h"
static PyMethodDef module_methods[] = {
    {"gen_urn", (PyCFunction)CPyPy_gen_urn, METH_VARARGS | METH_KEYWORDS, NULL /* docstring */},
    {"gen_ic_vote", (PyCFunction)CPyPy_gen_ic_vote, METH_VARARGS | METH_KEYWORDS, NULL /* docstring */},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "generate_urn",
    NULL, /* docstring */
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    module_methods
};

PyMODINIT_FUNC PyInit_generate_urn(void)
{
    if (CPyModule_generate_urn_internal) {
        Py_INCREF(CPyModule_generate_urn_internal);
        return CPyModule_generate_urn_internal;
    }
    CPyModule_generate_urn_internal = PyModule_Create(&module);
    if (unlikely(CPyModule_generate_urn_internal == NULL))
        return NULL;
    PyObject *modname = PyObject_GetAttrString((PyObject *)CPyModule_generate_urn_internal, "__name__");
    CPyStatic_globals = PyModule_GetDict(CPyModule_generate_urn_internal);
    if (unlikely(CPyStatic_globals == NULL))
        return NULL;
    if (CPyGlobalsInit() < 0)
        return NULL;
    char result = CPyDef___top_level__();
    if (result == 2)
        return NULL;
    Py_DECREF(modname);
    return CPyModule_generate_urn_internal;
}

PyObject *CPyDef_gen_urn(CPyTagged cpy_r_numvotes, CPyTagged cpy_r_replace, PyObject *cpy_r_alts) {
    PyObject *cpy_r_r0;
    PyObject *cpy_r_voteMap;
    PyObject *cpy_r_r1;
    PyObject *cpy_r_ReplaceVotes;
    PyObject *cpy_r_r2;
    PyObject *cpy_r_r3;
    PyObject *cpy_r_r4;
    CPyPtr cpy_r_r5;
    int64_t cpy_r_r6;
    CPyTagged cpy_r_r7;
    PyObject *cpy_r_r8;
    PyObject *cpy_r_r9;
    CPyTagged cpy_r_r10;
    CPyTagged cpy_r_ICsize;
    CPyTagged cpy_r_ReplaceSize;
    CPyTagged cpy_r_r11;
    CPyTagged cpy_r_x;
    int64_t cpy_r_r12;
    char cpy_r_r13;
    int64_t cpy_r_r14;
    char cpy_r_r15;
    char cpy_r_r16;
    char cpy_r_r17;
    char cpy_r_r18;
    char cpy_r_r19;
    PyObject *cpy_r_r20;
    PyObject *cpy_r_r21;
    PyObject *cpy_r_r22;
    CPyTagged cpy_r_r23;
    PyObject *cpy_r_r24;
    PyObject *cpy_r_r25;
    PyObject *cpy_r_r26;
    CPyTagged cpy_r_r27;
    CPyTagged cpy_r_flip;
    int64_t cpy_r_r28;
    char cpy_r_r29;
    int64_t cpy_r_r30;
    char cpy_r_r31;
    char cpy_r_r32;
    char cpy_r_r33;
    PyObject *cpy_r_r34;
    PyObject *cpy_r_tvote;
    PyObject *cpy_r_r35;
    PyObject *cpy_r_r36;
    CPyTagged cpy_r_r37;
    CPyTagged cpy_r_r38;
    PyObject *cpy_r_r39;
    int32_t cpy_r_r40;
    char cpy_r_r41;
    PyObject *cpy_r_r42;
    PyObject *cpy_r_r43;
    CPyTagged cpy_r_r44;
    CPyTagged cpy_r_r45;
    PyObject *cpy_r_r46;
    int32_t cpy_r_r47;
    char cpy_r_r48;
    CPyTagged cpy_r_r49;
    CPyTagged cpy_r_r50;
    CPyTagged cpy_r_r51;
    int64_t cpy_r_r52;
    CPyTagged cpy_r_r53;
    PyObject *cpy_r_r54;
    tuple_T3CIO cpy_r_r55;
    CPyTagged cpy_r_r56;
    char cpy_r_r57;
    PyObject *cpy_r_r58;
    PyObject *cpy_r_r59;
    PyObject *cpy_r_vote;
    PyObject *cpy_r_r60;
    CPyTagged cpy_r_r61;
    CPyTagged cpy_r_r62;
    int64_t cpy_r_r63;
    char cpy_r_r64;
    char cpy_r_r65;
    char cpy_r_r66;
    PyObject *cpy_r_r67;
    PyObject *cpy_r_r68;
    CPyTagged cpy_r_r69;
    CPyTagged cpy_r_r70;
    PyObject *cpy_r_r71;
    int32_t cpy_r_r72;
    char cpy_r_r73;
    PyObject *cpy_r_r74;
    PyObject *cpy_r_r75;
    CPyTagged cpy_r_r76;
    CPyTagged cpy_r_r77;
    PyObject *cpy_r_r78;
    int32_t cpy_r_r79;
    char cpy_r_r80;
    CPyTagged cpy_r_r81;
    char cpy_r_r82;
    char cpy_r_r83;
    PyObject *cpy_r_r84;
    PyObject *cpy_r_r85;
    PyObject *cpy_r_r86;
    PyObject *cpy_r_r87;
    PyObject *cpy_r_r88;
    PyObject *cpy_r_r89;
    PyObject *cpy_r_r90;
    PyObject *cpy_r_r91;
    PyObject *cpy_r_r92;
    CPyTagged cpy_r_r93;
    PyObject *cpy_r_r94;
CPyL0: ;
    cpy_r_r0 = PyDict_New();
    if (unlikely(cpy_r_r0 == NULL)) {
        CPy_AddTraceback("generate_urn.py", "gen_urn", 10, CPyStatic_globals);
        goto CPyL48;
    } else
        goto CPyL1;
CPyL1: ;
    cpy_r_voteMap = cpy_r_r0;
    cpy_r_r1 = PyDict_New();
    if (unlikely(cpy_r_r1 == NULL)) {
        CPy_AddTraceback("generate_urn.py", "gen_urn", 11, CPyStatic_globals);
        goto CPyL49;
    } else
        goto CPyL2;
CPyL2: ;
    cpy_r_ReplaceVotes = cpy_r_r1;
    cpy_r_r2 = CPyModule_math;
    cpy_r_r3 = CPyStatic_unicode_9; /* 'factorial' */
    cpy_r_r4 = CPyObject_GetAttr(cpy_r_r2, cpy_r_r3);
    if (unlikely(cpy_r_r4 == NULL)) {
        CPy_AddTraceback("generate_urn.py", "gen_urn", 13, CPyStatic_globals);
        goto CPyL50;
    } else
        goto CPyL3;
CPyL3: ;
    cpy_r_r5 = (CPyPtr)&((PyVarObject *)cpy_r_alts)->ob_size;
    cpy_r_r6 = *(int64_t *)cpy_r_r5;
    cpy_r_r7 = cpy_r_r6 << 1;
    cpy_r_r8 = CPyTagged_StealAsObject(cpy_r_r7);
    cpy_r_r9 = PyObject_CallFunctionObjArgs(cpy_r_r4, cpy_r_r8, NULL);
    CPy_DecRef(cpy_r_r4);
    CPy_DecRef(cpy_r_r8);
    if (unlikely(cpy_r_r9 == NULL)) {
        CPy_AddTraceback("generate_urn.py", "gen_urn", 13, CPyStatic_globals);
        goto CPyL50;
    } else
        goto CPyL4;
CPyL4: ;
    if (likely(PyLong_Check(cpy_r_r9)))
        cpy_r_r10 = CPyTagged_FromObject(cpy_r_r9);
    else {
        CPy_TypeError("int", cpy_r_r9);
        cpy_r_r10 = CPY_INT_TAG;
    }
    CPy_DecRef(cpy_r_r9);
    if (unlikely(cpy_r_r10 == CPY_INT_TAG)) {
        CPy_AddTraceback("generate_urn.py", "gen_urn", 13, CPyStatic_globals);
        goto CPyL50;
    } else
        goto CPyL5;
CPyL5: ;
    cpy_r_ICsize = cpy_r_r10;
    cpy_r_ReplaceSize = 0;
    cpy_r_r11 = 0;
    CPyTagged_IncRef(cpy_r_r11);
    cpy_r_x = cpy_r_r11;
    CPyTagged_DecRef(cpy_r_x);
    goto CPyL6;
CPyL6: ;
    cpy_r_r12 = cpy_r_r11 & 1;
    cpy_r_r13 = cpy_r_r12 == 0;
    cpy_r_r14 = cpy_r_numvotes & 1;
    cpy_r_r15 = cpy_r_r14 == 0;
    cpy_r_r16 = cpy_r_r13 & cpy_r_r15;
    if (cpy_r_r16) {
        goto CPyL7;
    } else
        goto CPyL8;
CPyL7: ;
    cpy_r_r17 = (Py_ssize_t)cpy_r_r11 < (Py_ssize_t)cpy_r_numvotes;
    cpy_r_r18 = cpy_r_r17;
    goto CPyL9;
CPyL8: ;
    cpy_r_r19 = CPyTagged_IsLt_(cpy_r_r11, cpy_r_numvotes);
    cpy_r_r18 = cpy_r_r19;
    goto CPyL9;
CPyL9: ;
    if (cpy_r_r18) {
        goto CPyL10;
    } else
        goto CPyL51;
CPyL10: ;
    cpy_r_r20 = CPyModule_random;
    cpy_r_r21 = CPyStatic_unicode_10; /* 'randint' */
    cpy_r_r22 = CPyObject_GetAttr(cpy_r_r20, cpy_r_r21);
    if (unlikely(cpy_r_r22 == NULL)) {
        CPy_AddTraceback("generate_urn.py", "gen_urn", 20, CPyStatic_globals);
        goto CPyL52;
    } else
        goto CPyL11;
CPyL11: ;
    cpy_r_r23 = CPyTagged_Add(cpy_r_ICsize, cpy_r_ReplaceSize);
    cpy_r_r24 = CPyTagged_StealAsObject(2);
    cpy_r_r25 = CPyTagged_StealAsObject(cpy_r_r23);
    cpy_r_r26 = PyObject_CallFunctionObjArgs(cpy_r_r22, cpy_r_r24, cpy_r_r25, NULL);
    CPy_DecRef(cpy_r_r22);
    CPy_DecRef(cpy_r_r24);
    CPy_DecRef(cpy_r_r25);
    if (unlikely(cpy_r_r26 == NULL)) {
        CPy_AddTraceback("generate_urn.py", "gen_urn", 20, CPyStatic_globals);
        goto CPyL52;
    } else
        goto CPyL12;
CPyL12: ;
    if (likely(PyLong_Check(cpy_r_r26)))
        cpy_r_r27 = CPyTagged_FromObject(cpy_r_r26);
    else {
        CPy_TypeError("int", cpy_r_r26);
        cpy_r_r27 = CPY_INT_TAG;
    }
    CPy_DecRef(cpy_r_r26);
    if (unlikely(cpy_r_r27 == CPY_INT_TAG)) {
        CPy_AddTraceback("generate_urn.py", "gen_urn", 20, CPyStatic_globals);
        goto CPyL52;
    } else
        goto CPyL13;
CPyL13: ;
    cpy_r_flip = cpy_r_r27;
    cpy_r_r28 = cpy_r_flip & 1;
    cpy_r_r29 = cpy_r_r28 != 0;
    if (cpy_r_r29) {
        goto CPyL15;
    } else
        goto CPyL14;
CPyL14: ;
    cpy_r_r30 = cpy_r_ICsize & 1;
    cpy_r_r31 = cpy_r_r30 != 0;
    if (cpy_r_r31) {
        goto CPyL15;
    } else
        goto CPyL16;
CPyL15: ;
    cpy_r_r32 = CPyTagged_IsLt_(cpy_r_ICsize, cpy_r_flip);
    if (cpy_r_r32) {
        goto CPyL25;
    } else
        goto CPyL53;
CPyL16: ;
    cpy_r_r33 = (Py_ssize_t)cpy_r_flip <= (Py_ssize_t)cpy_r_ICsize;
    if (cpy_r_r33) {
        goto CPyL53;
    } else
        goto CPyL25;
CPyL17: ;
    cpy_r_r34 = CPyDef_gen_ic_vote(cpy_r_alts);
    if (unlikely(cpy_r_r34 == NULL)) {
        CPy_AddTraceback("generate_urn.py", "gen_urn", 25, CPyStatic_globals);
        goto CPyL52;
    } else
        goto CPyL18;
CPyL18: ;
    cpy_r_tvote = cpy_r_r34;
    cpy_r_r35 = CPyTagged_StealAsObject(0);
    cpy_r_r36 = CPyDict_Get(cpy_r_voteMap, cpy_r_tvote, cpy_r_r35);
    CPy_DecRef(cpy_r_r35);
    if (unlikely(cpy_r_r36 == NULL)) {
        CPy_AddTraceback("generate_urn.py", "gen_urn", 26, CPyStatic_globals);
        goto CPyL54;
    } else
        goto CPyL19;
CPyL19: ;
    if (likely(PyLong_Check(cpy_r_r36)))
        cpy_r_r37 = CPyTagged_FromObject(cpy_r_r36);
    else {
        CPy_TypeError("int", cpy_r_r36);
        cpy_r_r37 = CPY_INT_TAG;
    }
    CPy_DecRef(cpy_r_r36);
    if (unlikely(cpy_r_r37 == CPY_INT_TAG)) {
        CPy_AddTraceback("generate_urn.py", "gen_urn", 26, CPyStatic_globals);
        goto CPyL54;
    } else
        goto CPyL20;
CPyL20: ;
    cpy_r_r38 = CPyTagged_Add(cpy_r_r37, 2);
    CPyTagged_DecRef(cpy_r_r37);
    cpy_r_r39 = CPyTagged_StealAsObject(cpy_r_r38);
    cpy_r_r40 = CPyDict_SetItem(cpy_r_voteMap, cpy_r_tvote, cpy_r_r39);
    CPy_DecRef(cpy_r_r39);
    cpy_r_r41 = cpy_r_r40 >= 0;
    if (unlikely(!cpy_r_r41)) {
        CPy_AddTraceback("generate_urn.py", "gen_urn", 26, CPyStatic_globals);
        goto CPyL54;
    } else
        goto CPyL21;
CPyL21: ;
    cpy_r_r42 = CPyTagged_StealAsObject(0);
    cpy_r_r43 = CPyDict_Get(cpy_r_ReplaceVotes, cpy_r_tvote, cpy_r_r42);
    CPy_DecRef(cpy_r_r42);
    if (unlikely(cpy_r_r43 == NULL)) {
        CPy_AddTraceback("generate_urn.py", "gen_urn", 27, CPyStatic_globals);
        goto CPyL54;
    } else
        goto CPyL22;
CPyL22: ;
    if (likely(PyLong_Check(cpy_r_r43)))
        cpy_r_r44 = CPyTagged_FromObject(cpy_r_r43);
    else {
        CPy_TypeError("int", cpy_r_r43);
        cpy_r_r44 = CPY_INT_TAG;
    }
    CPy_DecRef(cpy_r_r43);
    if (unlikely(cpy_r_r44 == CPY_INT_TAG)) {
        CPy_AddTraceback("generate_urn.py", "gen_urn", 27, CPyStatic_globals);
        goto CPyL54;
    } else
        goto CPyL23;
CPyL23: ;
    cpy_r_r45 = CPyTagged_Add(cpy_r_r44, cpy_r_replace);
    CPyTagged_DecRef(cpy_r_r44);
    cpy_r_r46 = CPyTagged_StealAsObject(cpy_r_r45);
    cpy_r_r47 = CPyDict_SetItem(cpy_r_ReplaceVotes, cpy_r_tvote, cpy_r_r46);
    CPy_DecRef(cpy_r_tvote);
    CPy_DecRef(cpy_r_r46);
    cpy_r_r48 = cpy_r_r47 >= 0;
    if (unlikely(!cpy_r_r48)) {
        CPy_AddTraceback("generate_urn.py", "gen_urn", 27, CPyStatic_globals);
        goto CPyL52;
    } else
        goto CPyL24;
CPyL24: ;
    cpy_r_r49 = CPyTagged_Add(cpy_r_ReplaceSize, cpy_r_replace);
    CPyTagged_DecRef(cpy_r_ReplaceSize);
    cpy_r_ReplaceSize = cpy_r_r49;
    goto CPyL46;
CPyL25: ;
    cpy_r_r50 = CPyTagged_Subtract(cpy_r_flip, cpy_r_ICsize);
    CPyTagged_DecRef(cpy_r_flip);
    cpy_r_flip = cpy_r_r50;
    cpy_r_r51 = 0;
    cpy_r_r52 = PyDict_Size(cpy_r_ReplaceVotes);
    cpy_r_r53 = cpy_r_r52 << 1;
    cpy_r_r54 = CPyDict_GetKeysIter(cpy_r_ReplaceVotes);
    if (unlikely(cpy_r_r54 == NULL)) {
        CPy_AddTraceback("generate_urn.py", "gen_urn", 35, CPyStatic_globals);
        goto CPyL55;
    } else
        goto CPyL26;
CPyL26: ;
    cpy_r_r55 = CPyDict_NextKey(cpy_r_r54, cpy_r_r51);
    cpy_r_r56 = cpy_r_r55.f1;
    CPyTagged_IncRef(cpy_r_r56);
    cpy_r_r51 = cpy_r_r56;
    cpy_r_r57 = cpy_r_r55.f0;
    if (cpy_r_r57) {
        goto CPyL27;
    } else
        goto CPyL56;
CPyL27: ;
    cpy_r_r58 = cpy_r_r55.f2;
    CPy_INCREF(cpy_r_r58);
    CPyTagged_DecRef(cpy_r_r55.f1);
    CPy_DecRef(cpy_r_r55.f2);
    if (likely(PyTuple_Check(cpy_r_r58)))
        cpy_r_r59 = cpy_r_r58;
    else {
        CPy_TypeError("tuple", cpy_r_r58);
        cpy_r_r59 = NULL;
    }
    if (unlikely(cpy_r_r59 == NULL)) {
        CPy_AddTraceback("generate_urn.py", "gen_urn", 35, CPyStatic_globals);
        goto CPyL57;
    } else
        goto CPyL28;
CPyL28: ;
    cpy_r_vote = cpy_r_r59;
    cpy_r_r60 = CPyDict_GetItem(cpy_r_ReplaceVotes, cpy_r_vote);
    if (unlikely(cpy_r_r60 == NULL)) {
        CPy_AddTraceback("generate_urn.py", "gen_urn", 37, CPyStatic_globals);
        goto CPyL58;
    } else
        goto CPyL29;
CPyL29: ;
    if (likely(PyLong_Check(cpy_r_r60)))
        cpy_r_r61 = CPyTagged_FromObject(cpy_r_r60);
    else {
        CPy_TypeError("int", cpy_r_r60);
        cpy_r_r61 = CPY_INT_TAG;
    }
    CPy_DecRef(cpy_r_r60);
    if (unlikely(cpy_r_r61 == CPY_INT_TAG)) {
        CPy_AddTraceback("generate_urn.py", "gen_urn", 37, CPyStatic_globals);
        goto CPyL58;
    } else
        goto CPyL30;
CPyL30: ;
    cpy_r_r62 = CPyTagged_Subtract(cpy_r_flip, cpy_r_r61);
    CPyTagged_DecRef(cpy_r_flip);
    CPyTagged_DecRef(cpy_r_r61);
    cpy_r_flip = cpy_r_r62;
    cpy_r_r63 = cpy_r_flip & 1;
    cpy_r_r64 = cpy_r_r63 != 0;
    if (cpy_r_r64) {
        goto CPyL31;
    } else
        goto CPyL32;
CPyL31: ;
    cpy_r_r65 = CPyTagged_IsLt_(0, cpy_r_flip);
    if (cpy_r_r65) {
        goto CPyL59;
    } else
        goto CPyL60;
CPyL32: ;
    cpy_r_r66 = (Py_ssize_t)cpy_r_flip <= (Py_ssize_t)0;
    if (cpy_r_r66) {
        goto CPyL60;
    } else
        goto CPyL59;
CPyL33: ;
    cpy_r_r67 = CPyTagged_StealAsObject(0);
    cpy_r_r68 = CPyDict_Get(cpy_r_voteMap, cpy_r_vote, cpy_r_r67);
    CPy_DecRef(cpy_r_r67);
    if (unlikely(cpy_r_r68 == NULL)) {
        CPy_AddTraceback("generate_urn.py", "gen_urn", 40, CPyStatic_globals);
        goto CPyL61;
    } else
        goto CPyL34;
CPyL34: ;
    if (likely(PyLong_Check(cpy_r_r68)))
        cpy_r_r69 = CPyTagged_FromObject(cpy_r_r68);
    else {
        CPy_TypeError("int", cpy_r_r68);
        cpy_r_r69 = CPY_INT_TAG;
    }
    CPy_DecRef(cpy_r_r68);
    if (unlikely(cpy_r_r69 == CPY_INT_TAG)) {
        CPy_AddTraceback("generate_urn.py", "gen_urn", 40, CPyStatic_globals);
        goto CPyL61;
    } else
        goto CPyL35;
CPyL35: ;
    cpy_r_r70 = CPyTagged_Add(cpy_r_r69, 2);
    CPyTagged_DecRef(cpy_r_r69);
    cpy_r_r71 = CPyTagged_StealAsObject(cpy_r_r70);
    cpy_r_r72 = CPyDict_SetItem(cpy_r_voteMap, cpy_r_vote, cpy_r_r71);
    CPy_DecRef(cpy_r_r71);
    cpy_r_r73 = cpy_r_r72 >= 0;
    if (unlikely(!cpy_r_r73)) {
        CPy_AddTraceback("generate_urn.py", "gen_urn", 40, CPyStatic_globals);
        goto CPyL61;
    } else
        goto CPyL36;
CPyL36: ;
    cpy_r_r74 = CPyTagged_StealAsObject(0);
    cpy_r_r75 = CPyDict_Get(cpy_r_ReplaceVotes, cpy_r_vote, cpy_r_r74);
    CPy_DecRef(cpy_r_r74);
    if (unlikely(cpy_r_r75 == NULL)) {
        CPy_AddTraceback("generate_urn.py", "gen_urn", 41, CPyStatic_globals);
        goto CPyL61;
    } else
        goto CPyL37;
CPyL37: ;
    if (likely(PyLong_Check(cpy_r_r75)))
        cpy_r_r76 = CPyTagged_FromObject(cpy_r_r75);
    else {
        CPy_TypeError("int", cpy_r_r75);
        cpy_r_r76 = CPY_INT_TAG;
    }
    CPy_DecRef(cpy_r_r75);
    if (unlikely(cpy_r_r76 == CPY_INT_TAG)) {
        CPy_AddTraceback("generate_urn.py", "gen_urn", 41, CPyStatic_globals);
        goto CPyL61;
    } else
        goto CPyL38;
CPyL38: ;
    cpy_r_r77 = CPyTagged_Add(cpy_r_r76, cpy_r_replace);
    CPyTagged_DecRef(cpy_r_r76);
    cpy_r_r78 = CPyTagged_StealAsObject(cpy_r_r77);
    cpy_r_r79 = CPyDict_SetItem(cpy_r_ReplaceVotes, cpy_r_vote, cpy_r_r78);
    CPy_DecRef(cpy_r_vote);
    CPy_DecRef(cpy_r_r78);
    cpy_r_r80 = cpy_r_r79 >= 0;
    if (unlikely(!cpy_r_r80)) {
        CPy_AddTraceback("generate_urn.py", "gen_urn", 41, CPyStatic_globals);
        goto CPyL52;
    } else
        goto CPyL39;
CPyL39: ;
    cpy_r_r81 = CPyTagged_Add(cpy_r_ReplaceSize, cpy_r_replace);
    CPyTagged_DecRef(cpy_r_ReplaceSize);
    cpy_r_ReplaceSize = cpy_r_r81;
    goto CPyL46;
CPyL40: ;
    cpy_r_r82 = CPyDict_CheckSize(cpy_r_ReplaceVotes, cpy_r_r53);
    if (unlikely(!cpy_r_r82)) {
        CPy_AddTraceback("generate_urn.py", "gen_urn", 35, CPyStatic_globals);
        goto CPyL57;
    } else
        goto CPyL26;
CPyL41: ;
    cpy_r_r83 = CPy_NoErrOccured();
    if (unlikely(!cpy_r_r83)) {
        CPy_AddTraceback("generate_urn.py", "gen_urn", 35, CPyStatic_globals);
        goto CPyL52;
    } else
        goto CPyL42;
CPyL42: ;
    cpy_r_r84 = CPyStatic_unicode_11; /* 'We Have a problem... replace fell through....' */
    cpy_r_r85 = CPyModule_builtins;
    cpy_r_r86 = CPyStatic_unicode_12; /* 'print' */
    cpy_r_r87 = CPyObject_GetAttr(cpy_r_r85, cpy_r_r86);
    if (unlikely(cpy_r_r87 == NULL)) {
        CPy_AddTraceback("generate_urn.py", "gen_urn", 45, CPyStatic_globals);
        goto CPyL52;
    } else
        goto CPyL43;
CPyL43: ;
    cpy_r_r88 = PyObject_CallFunctionObjArgs(cpy_r_r87, cpy_r_r84, NULL);
    CPy_DecRef(cpy_r_r87);
    if (unlikely(cpy_r_r88 == NULL)) {
        CPy_AddTraceback("generate_urn.py", "gen_urn", 45, CPyStatic_globals);
        goto CPyL52;
    } else
        goto CPyL62;
CPyL44: ;
    cpy_r_r89 = CPyModule_builtins;
    cpy_r_r90 = CPyStatic_unicode_13; /* 'exit' */
    cpy_r_r91 = CPyObject_GetAttr(cpy_r_r89, cpy_r_r90);
    if (unlikely(cpy_r_r91 == NULL)) {
        CPy_AddTraceback("generate_urn.py", "gen_urn", 46, CPyStatic_globals);
        goto CPyL52;
    } else
        goto CPyL45;
CPyL45: ;
    cpy_r_r92 = PyObject_CallFunctionObjArgs(cpy_r_r91, NULL);
    CPy_DecRef(cpy_r_r91);
    if (unlikely(cpy_r_r92 == NULL)) {
        CPy_AddTraceback("generate_urn.py", "gen_urn", 46, CPyStatic_globals);
        goto CPyL52;
    } else
        goto CPyL63;
CPyL46: ;
    cpy_r_r93 = CPyTagged_Add(cpy_r_r11, 2);
    CPyTagged_DecRef(cpy_r_r11);
    CPyTagged_IncRef(cpy_r_r93);
    cpy_r_r11 = cpy_r_r93;
    cpy_r_x = cpy_r_r93;
    CPyTagged_DecRef(cpy_r_x);
    goto CPyL6;
CPyL47: ;
    return cpy_r_voteMap;
CPyL48: ;
    cpy_r_r94 = NULL;
    return cpy_r_r94;
CPyL49: ;
    CPy_DecRef(cpy_r_voteMap);
    goto CPyL48;
CPyL50: ;
    CPy_DecRef(cpy_r_voteMap);
    CPy_DecRef(cpy_r_ReplaceVotes);
    goto CPyL48;
CPyL51: ;
    CPy_DecRef(cpy_r_ReplaceVotes);
    CPyTagged_DecRef(cpy_r_ICsize);
    CPyTagged_DecRef(cpy_r_ReplaceSize);
    CPyTagged_DecRef(cpy_r_r11);
    goto CPyL47;
CPyL52: ;
    CPy_DecRef(cpy_r_voteMap);
    CPy_DecRef(cpy_r_ReplaceVotes);
    CPyTagged_DecRef(cpy_r_ICsize);
    CPyTagged_DecRef(cpy_r_ReplaceSize);
    CPyTagged_DecRef(cpy_r_r11);
    goto CPyL48;
CPyL53: ;
    CPyTagged_DecRef(cpy_r_flip);
    goto CPyL17;
CPyL54: ;
    CPy_DecRef(cpy_r_voteMap);
    CPy_DecRef(cpy_r_ReplaceVotes);
    CPyTagged_DecRef(cpy_r_ICsize);
    CPyTagged_DecRef(cpy_r_ReplaceSize);
    CPyTagged_DecRef(cpy_r_r11);
    CPy_DecRef(cpy_r_tvote);
    goto CPyL48;
CPyL55: ;
    CPy_DecRef(cpy_r_voteMap);
    CPy_DecRef(cpy_r_ReplaceVotes);
    CPyTagged_DecRef(cpy_r_ICsize);
    CPyTagged_DecRef(cpy_r_ReplaceSize);
    CPyTagged_DecRef(cpy_r_r11);
    CPyTagged_DecRef(cpy_r_flip);
    goto CPyL48;
CPyL56: ;
    CPyTagged_DecRef(cpy_r_flip);
    CPy_DecRef(cpy_r_r54);
    CPyTagged_DecRef(cpy_r_r55.f1);
    CPy_DecRef(cpy_r_r55.f2);
    goto CPyL41;
CPyL57: ;
    CPy_DecRef(cpy_r_voteMap);
    CPy_DecRef(cpy_r_ReplaceVotes);
    CPyTagged_DecRef(cpy_r_ICsize);
    CPyTagged_DecRef(cpy_r_ReplaceSize);
    CPyTagged_DecRef(cpy_r_r11);
    CPyTagged_DecRef(cpy_r_flip);
    CPy_DecRef(cpy_r_r54);
    goto CPyL48;
CPyL58: ;
    CPy_DecRef(cpy_r_voteMap);
    CPy_DecRef(cpy_r_ReplaceVotes);
    CPyTagged_DecRef(cpy_r_ICsize);
    CPyTagged_DecRef(cpy_r_ReplaceSize);
    CPyTagged_DecRef(cpy_r_r11);
    CPyTagged_DecRef(cpy_r_flip);
    CPy_DecRef(cpy_r_r54);
    CPy_DecRef(cpy_r_vote);
    goto CPyL48;
CPyL59: ;
    CPy_DecRef(cpy_r_vote);
    goto CPyL40;
CPyL60: ;
    CPyTagged_DecRef(cpy_r_flip);
    CPy_DecRef(cpy_r_r54);
    goto CPyL33;
CPyL61: ;
    CPy_DecRef(cpy_r_voteMap);
    CPy_DecRef(cpy_r_ReplaceVotes);
    CPyTagged_DecRef(cpy_r_ICsize);
    CPyTagged_DecRef(cpy_r_ReplaceSize);
    CPyTagged_DecRef(cpy_r_r11);
    CPy_DecRef(cpy_r_vote);
    goto CPyL48;
CPyL62: ;
    CPy_DecRef(cpy_r_r88);
    goto CPyL44;
CPyL63: ;
    CPy_DecRef(cpy_r_r92);
    goto CPyL46;
}

PyObject *CPyPy_gen_urn(PyObject *self, PyObject *args, PyObject *kw) {
    static char *kwlist[] = {"numvotes", "replace", "alts", 0};
    PyObject *obj_numvotes;
    PyObject *obj_replace;
    PyObject *obj_alts;
    if (!CPyArg_ParseTupleAndKeywords(args, kw, "OOO:gen_urn", kwlist, &obj_numvotes, &obj_replace, &obj_alts)) {
        return NULL;
    }
    CPyTagged arg_numvotes;
    if (likely(PyLong_Check(obj_numvotes)))
        arg_numvotes = CPyTagged_BorrowFromObject(obj_numvotes);
    else {
        CPy_TypeError("int", obj_numvotes);
        goto fail;
    }
    CPyTagged arg_replace;
    if (likely(PyLong_Check(obj_replace)))
        arg_replace = CPyTagged_BorrowFromObject(obj_replace);
    else {
        CPy_TypeError("int", obj_replace);
        goto fail;
    }
    PyObject *arg_alts;
    if (likely(PyList_Check(obj_alts)))
        arg_alts = obj_alts;
    else {
        CPy_TypeError("list", obj_alts);
        arg_alts = NULL;
    }
    if (arg_alts == NULL) goto fail;
    PyObject *retval = CPyDef_gen_urn(arg_numvotes, arg_replace, arg_alts);
    return retval;
fail: ;
    CPy_AddTraceback("generate_urn.py", "gen_urn", 9, CPyStatic_globals);
    return NULL;
}

PyObject *CPyDef_gen_ic_vote(PyObject *cpy_r_alts) {
    PyObject *cpy_r_r0;
    PyObject *cpy_r_options;
    PyObject *cpy_r_r1;
    PyObject *cpy_r_vote;
    CPyPtr cpy_r_r2;
    int64_t cpy_r_r3;
    CPyTagged cpy_r_r4;
    char cpy_r_r5;
    PyObject *cpy_r_r6;
    PyObject *cpy_r_r7;
    PyObject *cpy_r_r8;
    CPyPtr cpy_r_r9;
    int64_t cpy_r_r10;
    CPyTagged cpy_r_r11;
    CPyTagged cpy_r_r12;
    PyObject *cpy_r_r13;
    PyObject *cpy_r_r14;
    PyObject *cpy_r_r15;
    CPyTagged cpy_r_r16;
    PyObject *cpy_r_r17;
    CPyTagged cpy_r_r18;
    PyObject *cpy_r_r19;
    int32_t cpy_r_r20;
    char cpy_r_r21;
    PyObject *cpy_r_r22;
    PyObject *cpy_r_r23;
CPyL0: ;
    cpy_r_r0 = PySequence_List(cpy_r_alts);
    if (unlikely(cpy_r_r0 == NULL)) {
        CPy_AddTraceback("generate_urn.py", "gen_ic_vote", 55, CPyStatic_globals);
        goto CPyL12;
    } else
        goto CPyL1;
CPyL1: ;
    cpy_r_options = cpy_r_r0;
    cpy_r_r1 = PyList_New(0);
    if (unlikely(cpy_r_r1 == NULL)) {
        CPy_AddTraceback("generate_urn.py", "gen_ic_vote", 56, CPyStatic_globals);
        goto CPyL13;
    } else
        goto CPyL2;
CPyL2: ;
    cpy_r_vote = cpy_r_r1;
    goto CPyL3;
CPyL3: ;
    cpy_r_r2 = (CPyPtr)&((PyVarObject *)cpy_r_options)->ob_size;
    cpy_r_r3 = *(int64_t *)cpy_r_r2;
    cpy_r_r4 = cpy_r_r3 << 1;
    cpy_r_r5 = (Py_ssize_t)cpy_r_r4 > (Py_ssize_t)0;
    if (cpy_r_r5) {
        goto CPyL4;
    } else
        goto CPyL14;
CPyL4: ;
    cpy_r_r6 = CPyModule_random;
    cpy_r_r7 = CPyStatic_unicode_10; /* 'randint' */
    cpy_r_r8 = CPyObject_GetAttr(cpy_r_r6, cpy_r_r7);
    if (unlikely(cpy_r_r8 == NULL)) {
        CPy_AddTraceback("generate_urn.py", "gen_ic_vote", 59, CPyStatic_globals);
        goto CPyL15;
    } else
        goto CPyL5;
CPyL5: ;
    cpy_r_r9 = (CPyPtr)&((PyVarObject *)cpy_r_options)->ob_size;
    cpy_r_r10 = *(int64_t *)cpy_r_r9;
    cpy_r_r11 = cpy_r_r10 << 1;
    cpy_r_r12 = CPyTagged_Subtract(cpy_r_r11, 2);
    cpy_r_r13 = CPyTagged_StealAsObject(0);
    cpy_r_r14 = CPyTagged_StealAsObject(cpy_r_r12);
    cpy_r_r15 = PyObject_CallFunctionObjArgs(cpy_r_r8, cpy_r_r13, cpy_r_r14, NULL);
    CPy_DecRef(cpy_r_r8);
    CPy_DecRef(cpy_r_r13);
    CPy_DecRef(cpy_r_r14);
    if (unlikely(cpy_r_r15 == NULL)) {
        CPy_AddTraceback("generate_urn.py", "gen_ic_vote", 59, CPyStatic_globals);
        goto CPyL15;
    } else
        goto CPyL6;
CPyL6: ;
    if (likely(PyLong_Check(cpy_r_r15)))
        cpy_r_r16 = CPyTagged_FromObject(cpy_r_r15);
    else {
        CPy_TypeError("int", cpy_r_r15);
        cpy_r_r16 = CPY_INT_TAG;
    }
    CPy_DecRef(cpy_r_r15);
    if (unlikely(cpy_r_r16 == CPY_INT_TAG)) {
        CPy_AddTraceback("generate_urn.py", "gen_ic_vote", 59, CPyStatic_globals);
        goto CPyL15;
    } else
        goto CPyL7;
CPyL7: ;
    cpy_r_r17 = CPyList_Pop(cpy_r_options, cpy_r_r16);
    CPyTagged_DecRef(cpy_r_r16);
    if (unlikely(cpy_r_r17 == NULL)) {
        CPy_AddTraceback("generate_urn.py", "gen_ic_vote", 59, CPyStatic_globals);
        goto CPyL15;
    } else
        goto CPyL8;
CPyL8: ;
    if (likely(PyLong_Check(cpy_r_r17)))
        cpy_r_r18 = CPyTagged_FromObject(cpy_r_r17);
    else {
        CPy_TypeError("int", cpy_r_r17);
        cpy_r_r18 = CPY_INT_TAG;
    }
    CPy_DecRef(cpy_r_r17);
    if (unlikely(cpy_r_r18 == CPY_INT_TAG)) {
        CPy_AddTraceback("generate_urn.py", "gen_ic_vote", 59, CPyStatic_globals);
        goto CPyL15;
    } else
        goto CPyL9;
CPyL9: ;
    cpy_r_r19 = CPyTagged_StealAsObject(cpy_r_r18);
    cpy_r_r20 = PyList_Append(cpy_r_vote, cpy_r_r19);
    CPy_DecRef(cpy_r_r19);
    cpy_r_r21 = cpy_r_r20 >= 0;
    if (unlikely(!cpy_r_r21)) {
        CPy_AddTraceback("generate_urn.py", "gen_ic_vote", 59, CPyStatic_globals);
        goto CPyL15;
    } else
        goto CPyL3;
CPyL10: ;
    cpy_r_r22 = PyList_AsTuple(cpy_r_vote);
    CPy_DecRef(cpy_r_vote);
    if (unlikely(cpy_r_r22 == NULL)) {
        CPy_AddTraceback("generate_urn.py", "gen_ic_vote", 60, CPyStatic_globals);
        goto CPyL12;
    } else
        goto CPyL11;
CPyL11: ;
    return cpy_r_r22;
CPyL12: ;
    cpy_r_r23 = NULL;
    return cpy_r_r23;
CPyL13: ;
    CPy_DecRef(cpy_r_options);
    goto CPyL12;
CPyL14: ;
    CPy_DecRef(cpy_r_options);
    goto CPyL10;
CPyL15: ;
    CPy_DecRef(cpy_r_options);
    CPy_DecRef(cpy_r_vote);
    goto CPyL12;
}

PyObject *CPyPy_gen_ic_vote(PyObject *self, PyObject *args, PyObject *kw) {
    static char *kwlist[] = {"alts", 0};
    PyObject *obj_alts;
    if (!CPyArg_ParseTupleAndKeywords(args, kw, "O:gen_ic_vote", kwlist, &obj_alts)) {
        return NULL;
    }
    PyObject *arg_alts;
    if (likely(PyList_Check(obj_alts)))
        arg_alts = obj_alts;
    else {
        CPy_TypeError("list", obj_alts);
        arg_alts = NULL;
    }
    if (arg_alts == NULL) goto fail;
    PyObject *retval = CPyDef_gen_ic_vote(arg_alts);
    return retval;
fail: ;
    CPy_AddTraceback("generate_urn.py", "gen_ic_vote", 54, CPyStatic_globals);
    return NULL;
}

char CPyDef___top_level__(void) {
    PyObject *cpy_r_r0;
    PyObject *cpy_r_r1;
    char cpy_r_r2;
    PyObject *cpy_r_r3;
    PyObject *cpy_r_r4;
    PyObject *cpy_r_r5;
    PyObject *cpy_r_r6;
    char cpy_r_r7;
    PyObject *cpy_r_r8;
    PyObject *cpy_r_r9;
    PyObject *cpy_r_r10;
    PyObject *cpy_r_r11;
    PyObject *cpy_r_r12;
    PyObject *cpy_r_r13;
    PyObject *cpy_r_r14;
    int32_t cpy_r_r15;
    char cpy_r_r16;
    PyObject *cpy_r_r17;
    PyObject *cpy_r_r18;
    PyObject *cpy_r_r19;
    int32_t cpy_r_r20;
    char cpy_r_r21;
    PyObject *cpy_r_r22;
    PyObject *cpy_r_r23;
    PyObject *cpy_r_r24;
    int32_t cpy_r_r25;
    char cpy_r_r26;
    PyObject *cpy_r_r27;
    PyObject *cpy_r_r28;
    PyObject *cpy_r_r29;
    int32_t cpy_r_r30;
    char cpy_r_r31;
    PyObject *cpy_r_r32;
    PyObject *cpy_r_r33;
    PyObject *cpy_r_r34;
    char cpy_r_r35;
    PyObject *cpy_r_r36;
    PyObject *cpy_r_r37;
    PyObject *cpy_r_r38;
    PyObject *cpy_r_r39;
    PyObject *cpy_r_r40;
    PyObject *cpy_r_r41;
    int32_t cpy_r_r42;
    char cpy_r_r43;
    PyObject *cpy_r_r44;
    PyObject *cpy_r_r45;
    PyObject *cpy_r_r46;
    char cpy_r_r47;
    PyObject *cpy_r_r48;
    PyObject *cpy_r_r49;
    PyObject *cpy_r_r50;
    PyObject *cpy_r_r51;
    PyObject *cpy_r_r52;
    PyObject *cpy_r_r53;
    int32_t cpy_r_r54;
    char cpy_r_r55;
    PyObject *cpy_r_r56;
    PyObject *cpy_r_r57;
    PyObject *cpy_r_r58;
    char cpy_r_r59;
    PyObject *cpy_r_r60;
    PyObject *cpy_r_r61;
    PyObject *cpy_r_r62;
    PyObject *cpy_r_r63;
    PyObject *cpy_r_r64;
    PyObject *cpy_r_r65;
    int32_t cpy_r_r66;
    char cpy_r_r67;
    PyObject *cpy_r_r68;
    PyObject *cpy_r_r69;
    PyObject *cpy_r_r70;
    PyObject *cpy_r_r71;
    PyObject *cpy_r_r72;
    PyObject *cpy_r_r73;
    PyObject *cpy_r_r74;
    int32_t cpy_r_r75;
    char cpy_r_r76;
    PyObject *cpy_r_r77;
    PyObject *cpy_r_r78;
    PyObject *cpy_r_r79;
    PyObject *cpy_r_r80;
    PyObject *cpy_r_r81;
    CPyPtr cpy_r_r82;
    CPyPtr cpy_r_r83;
    CPyPtr cpy_r_r84;
    CPyPtr cpy_r_r85;
    CPyPtr cpy_r_r86;
    PyObject *cpy_r_r87;
    PyObject *cpy_r_r88;
    PyObject *cpy_r_r89;
    PyObject *cpy_r_r90;
    PyObject *cpy_r_r91;
    PyObject *cpy_r_r92;
    PyObject *cpy_r_r93;
    PyObject *cpy_r_r94;
    PyObject *cpy_r_r95;
    PyObject *cpy_r_r96;
    PyObject *cpy_r_r97;
    PyObject *cpy_r_r98;
    PyObject *cpy_r_r99;
    PyObject *cpy_r_r100;
    PyObject *cpy_r_r101;
    PyObject *cpy_r_r102;
    PyObject *cpy_r_r103;
    PyObject *cpy_r_r104;
    PyObject *cpy_r_r105;
    PyObject *cpy_r_r106;
    char cpy_r_r107;
CPyL0: ;
    cpy_r_r0 = CPyModule_builtins;
    cpy_r_r1 = (PyObject *)&_Py_NoneStruct;
    cpy_r_r2 = cpy_r_r0 != cpy_r_r1;
    if (cpy_r_r2) {
        goto CPyL3;
    } else
        goto CPyL1;
CPyL1: ;
    cpy_r_r3 = CPyStatic_unicode_0; /* 'builtins' */
    cpy_r_r4 = PyImport_Import(cpy_r_r3);
    if (unlikely(cpy_r_r4 == NULL)) {
        CPy_AddTraceback("generate_urn.py", "<module>", -1, CPyStatic_globals);
        goto CPyL47;
    } else
        goto CPyL2;
CPyL2: ;
    CPyModule_builtins = cpy_r_r4;
    CPy_INCREF(CPyModule_builtins);
    CPy_DecRef(cpy_r_r4);
    goto CPyL3;
CPyL3: ;
    cpy_r_r5 = CPyModule_typing;
    cpy_r_r6 = (PyObject *)&_Py_NoneStruct;
    cpy_r_r7 = cpy_r_r5 != cpy_r_r6;
    if (cpy_r_r7) {
        goto CPyL6;
    } else
        goto CPyL4;
CPyL4: ;
    cpy_r_r8 = CPyStatic_unicode_1; /* 'typing' */
    cpy_r_r9 = PyImport_Import(cpy_r_r8);
    if (unlikely(cpy_r_r9 == NULL)) {
        CPy_AddTraceback("generate_urn.py", "<module>", 1, CPyStatic_globals);
        goto CPyL47;
    } else
        goto CPyL5;
CPyL5: ;
    CPyModule_typing = cpy_r_r9;
    CPy_INCREF(CPyModule_typing);
    CPy_DecRef(cpy_r_r9);
    goto CPyL6;
CPyL6: ;
    cpy_r_r10 = CPyModule_typing;
    cpy_r_r11 = CPyStatic_globals;
    cpy_r_r12 = CPyStatic_unicode_2; /* 'List' */
    cpy_r_r13 = CPyObject_GetAttr(cpy_r_r10, cpy_r_r12);
    if (unlikely(cpy_r_r13 == NULL)) {
        CPy_AddTraceback("generate_urn.py", "<module>", 1, CPyStatic_globals);
        goto CPyL47;
    } else
        goto CPyL7;
CPyL7: ;
    cpy_r_r14 = CPyStatic_unicode_2; /* 'List' */
    cpy_r_r15 = CPyDict_SetItem(cpy_r_r11, cpy_r_r14, cpy_r_r13);
    CPy_DecRef(cpy_r_r13);
    cpy_r_r16 = cpy_r_r15 >= 0;
    if (unlikely(!cpy_r_r16)) {
        CPy_AddTraceback("generate_urn.py", "<module>", 1, CPyStatic_globals);
        goto CPyL47;
    } else
        goto CPyL8;
CPyL8: ;
    cpy_r_r17 = CPyStatic_unicode_3; /* 'Any' */
    cpy_r_r18 = CPyObject_GetAttr(cpy_r_r10, cpy_r_r17);
    if (unlikely(cpy_r_r18 == NULL)) {
        CPy_AddTraceback("generate_urn.py", "<module>", 1, CPyStatic_globals);
        goto CPyL47;
    } else
        goto CPyL9;
CPyL9: ;
    cpy_r_r19 = CPyStatic_unicode_3; /* 'Any' */
    cpy_r_r20 = CPyDict_SetItem(cpy_r_r11, cpy_r_r19, cpy_r_r18);
    CPy_DecRef(cpy_r_r18);
    cpy_r_r21 = cpy_r_r20 >= 0;
    if (unlikely(!cpy_r_r21)) {
        CPy_AddTraceback("generate_urn.py", "<module>", 1, CPyStatic_globals);
        goto CPyL47;
    } else
        goto CPyL10;
CPyL10: ;
    cpy_r_r22 = CPyStatic_unicode_4; /* 'Dict' */
    cpy_r_r23 = CPyObject_GetAttr(cpy_r_r10, cpy_r_r22);
    if (unlikely(cpy_r_r23 == NULL)) {
        CPy_AddTraceback("generate_urn.py", "<module>", 1, CPyStatic_globals);
        goto CPyL47;
    } else
        goto CPyL11;
CPyL11: ;
    cpy_r_r24 = CPyStatic_unicode_4; /* 'Dict' */
    cpy_r_r25 = CPyDict_SetItem(cpy_r_r11, cpy_r_r24, cpy_r_r23);
    CPy_DecRef(cpy_r_r23);
    cpy_r_r26 = cpy_r_r25 >= 0;
    if (unlikely(!cpy_r_r26)) {
        CPy_AddTraceback("generate_urn.py", "<module>", 1, CPyStatic_globals);
        goto CPyL47;
    } else
        goto CPyL12;
CPyL12: ;
    cpy_r_r27 = CPyStatic_unicode_5; /* 'Tuple' */
    cpy_r_r28 = CPyObject_GetAttr(cpy_r_r10, cpy_r_r27);
    if (unlikely(cpy_r_r28 == NULL)) {
        CPy_AddTraceback("generate_urn.py", "<module>", 1, CPyStatic_globals);
        goto CPyL47;
    } else
        goto CPyL13;
CPyL13: ;
    cpy_r_r29 = CPyStatic_unicode_5; /* 'Tuple' */
    cpy_r_r30 = CPyDict_SetItem(cpy_r_r11, cpy_r_r29, cpy_r_r28);
    CPy_DecRef(cpy_r_r28);
    cpy_r_r31 = cpy_r_r30 >= 0;
    if (unlikely(!cpy_r_r31)) {
        CPy_AddTraceback("generate_urn.py", "<module>", 1, CPyStatic_globals);
        goto CPyL47;
    } else
        goto CPyL14;
CPyL14: ;
    cpy_r_r32 = CPyStatic_globals;
    cpy_r_r33 = CPyModule_math;
    cpy_r_r34 = (PyObject *)&_Py_NoneStruct;
    cpy_r_r35 = cpy_r_r33 != cpy_r_r34;
    if (cpy_r_r35) {
        goto CPyL17;
    } else
        goto CPyL15;
CPyL15: ;
    cpy_r_r36 = CPyStatic_unicode_6; /* 'math' */
    cpy_r_r37 = PyImport_Import(cpy_r_r36);
    if (unlikely(cpy_r_r37 == NULL)) {
        CPy_AddTraceback("generate_urn.py", "<module>", 3, CPyStatic_globals);
        goto CPyL47;
    } else
        goto CPyL16;
CPyL16: ;
    CPyModule_math = cpy_r_r37;
    CPy_INCREF(CPyModule_math);
    CPy_DecRef(cpy_r_r37);
    goto CPyL17;
CPyL17: ;
    cpy_r_r38 = PyImport_GetModuleDict();
    cpy_r_r39 = CPyStatic_unicode_6; /* 'math' */
    cpy_r_r40 = CPyDict_GetItem(cpy_r_r38, cpy_r_r39);
    if (unlikely(cpy_r_r40 == NULL)) {
        CPy_AddTraceback("generate_urn.py", "<module>", 3, CPyStatic_globals);
        goto CPyL47;
    } else
        goto CPyL18;
CPyL18: ;
    cpy_r_r41 = CPyStatic_unicode_6; /* 'math' */
    cpy_r_r42 = CPyDict_SetItem(cpy_r_r32, cpy_r_r41, cpy_r_r40);
    CPy_DecRef(cpy_r_r40);
    cpy_r_r43 = cpy_r_r42 >= 0;
    if (unlikely(!cpy_r_r43)) {
        CPy_AddTraceback("generate_urn.py", "<module>", 3, CPyStatic_globals);
        goto CPyL47;
    } else
        goto CPyL19;
CPyL19: ;
    cpy_r_r44 = CPyStatic_globals;
    cpy_r_r45 = CPyModule_random;
    cpy_r_r46 = (PyObject *)&_Py_NoneStruct;
    cpy_r_r47 = cpy_r_r45 != cpy_r_r46;
    if (cpy_r_r47) {
        goto CPyL22;
    } else
        goto CPyL20;
CPyL20: ;
    cpy_r_r48 = CPyStatic_unicode_7; /* 'random' */
    cpy_r_r49 = PyImport_Import(cpy_r_r48);
    if (unlikely(cpy_r_r49 == NULL)) {
        CPy_AddTraceback("generate_urn.py", "<module>", 4, CPyStatic_globals);
        goto CPyL47;
    } else
        goto CPyL21;
CPyL21: ;
    CPyModule_random = cpy_r_r49;
    CPy_INCREF(CPyModule_random);
    CPy_DecRef(cpy_r_r49);
    goto CPyL22;
CPyL22: ;
    cpy_r_r50 = PyImport_GetModuleDict();
    cpy_r_r51 = CPyStatic_unicode_7; /* 'random' */
    cpy_r_r52 = CPyDict_GetItem(cpy_r_r50, cpy_r_r51);
    if (unlikely(cpy_r_r52 == NULL)) {
        CPy_AddTraceback("generate_urn.py", "<module>", 4, CPyStatic_globals);
        goto CPyL47;
    } else
        goto CPyL23;
CPyL23: ;
    cpy_r_r53 = CPyStatic_unicode_7; /* 'random' */
    cpy_r_r54 = CPyDict_SetItem(cpy_r_r44, cpy_r_r53, cpy_r_r52);
    CPy_DecRef(cpy_r_r52);
    cpy_r_r55 = cpy_r_r54 >= 0;
    if (unlikely(!cpy_r_r55)) {
        CPy_AddTraceback("generate_urn.py", "<module>", 4, CPyStatic_globals);
        goto CPyL47;
    } else
        goto CPyL24;
CPyL24: ;
    cpy_r_r56 = CPyStatic_globals;
    cpy_r_r57 = CPyModule_time;
    cpy_r_r58 = (PyObject *)&_Py_NoneStruct;
    cpy_r_r59 = cpy_r_r57 != cpy_r_r58;
    if (cpy_r_r59) {
        goto CPyL27;
    } else
        goto CPyL25;
CPyL25: ;
    cpy_r_r60 = CPyStatic_unicode_8; /* 'time' */
    cpy_r_r61 = PyImport_Import(cpy_r_r60);
    if (unlikely(cpy_r_r61 == NULL)) {
        CPy_AddTraceback("generate_urn.py", "<module>", 5, CPyStatic_globals);
        goto CPyL47;
    } else
        goto CPyL26;
CPyL26: ;
    CPyModule_time = cpy_r_r61;
    CPy_INCREF(CPyModule_time);
    CPy_DecRef(cpy_r_r61);
    goto CPyL27;
CPyL27: ;
    cpy_r_r62 = PyImport_GetModuleDict();
    cpy_r_r63 = CPyStatic_unicode_8; /* 'time' */
    cpy_r_r64 = CPyDict_GetItem(cpy_r_r62, cpy_r_r63);
    if (unlikely(cpy_r_r64 == NULL)) {
        CPy_AddTraceback("generate_urn.py", "<module>", 5, CPyStatic_globals);
        goto CPyL47;
    } else
        goto CPyL28;
CPyL28: ;
    cpy_r_r65 = CPyStatic_unicode_8; /* 'time' */
    cpy_r_r66 = CPyDict_SetItem(cpy_r_r56, cpy_r_r65, cpy_r_r64);
    CPy_DecRef(cpy_r_r64);
    cpy_r_r67 = cpy_r_r66 >= 0;
    if (unlikely(!cpy_r_r67)) {
        CPy_AddTraceback("generate_urn.py", "<module>", 5, CPyStatic_globals);
        goto CPyL47;
    } else
        goto CPyL29;
CPyL29: ;
    cpy_r_r68 = CPyModule_time;
    cpy_r_r69 = CPyStatic_unicode_8; /* 'time' */
    cpy_r_r70 = CPyObject_GetAttr(cpy_r_r68, cpy_r_r69);
    if (unlikely(cpy_r_r70 == NULL)) {
        CPy_AddTraceback("generate_urn.py", "<module>", 63, CPyStatic_globals);
        goto CPyL47;
    } else
        goto CPyL30;
CPyL30: ;
    cpy_r_r71 = PyObject_CallFunctionObjArgs(cpy_r_r70, NULL);
    CPy_DecRef(cpy_r_r70);
    if (unlikely(cpy_r_r71 == NULL)) {
        CPy_AddTraceback("generate_urn.py", "<module>", 63, CPyStatic_globals);
        goto CPyL47;
    } else
        goto CPyL31;
CPyL31: ;
    if (likely(CPyFloat_Check(cpy_r_r71)))
        cpy_r_r72 = cpy_r_r71;
    else {
        CPy_TypeError("float", cpy_r_r71);
        cpy_r_r72 = NULL;
    }
    if (unlikely(cpy_r_r72 == NULL)) {
        CPy_AddTraceback("generate_urn.py", "<module>", 63, CPyStatic_globals);
        goto CPyL47;
    } else
        goto CPyL32;
CPyL32: ;
    cpy_r_r73 = CPyStatic_globals;
    cpy_r_r74 = CPyStatic_unicode_14; /* 't0' */
    cpy_r_r75 = CPyDict_SetItem(cpy_r_r73, cpy_r_r74, cpy_r_r72);
    CPy_DecRef(cpy_r_r72);
    cpy_r_r76 = cpy_r_r75 >= 0;
    if (unlikely(!cpy_r_r76)) {
        CPy_AddTraceback("generate_urn.py", "<module>", 63, CPyStatic_globals);
        goto CPyL47;
    } else
        goto CPyL33;
CPyL33: ;
    cpy_r_r77 = PyList_New(4);
    if (unlikely(cpy_r_r77 == NULL)) {
        CPy_AddTraceback("generate_urn.py", "<module>", 64, CPyStatic_globals);
        goto CPyL47;
    } else
        goto CPyL34;
CPyL34: ;
    cpy_r_r78 = CPyTagged_StealAsObject(0);
    cpy_r_r79 = CPyTagged_StealAsObject(2);
    cpy_r_r80 = CPyTagged_StealAsObject(4);
    cpy_r_r81 = CPyTagged_StealAsObject(6);
    cpy_r_r82 = (CPyPtr)&((PyListObject *)cpy_r_r77)->ob_item;
    cpy_r_r83 = *(CPyPtr *)cpy_r_r82;
    *(PyObject * *)cpy_r_r83 = cpy_r_r78;
    cpy_r_r84 = cpy_r_r83 + 8;
    *(PyObject * *)cpy_r_r84 = cpy_r_r79;
    cpy_r_r85 = cpy_r_r83 + 16;
    *(PyObject * *)cpy_r_r85 = cpy_r_r80;
    cpy_r_r86 = cpy_r_r83 + 24;
    *(PyObject * *)cpy_r_r86 = cpy_r_r81;
    cpy_r_r87 = CPyDef_gen_urn(20, 10, cpy_r_r77);
    CPy_DecRef(cpy_r_r77);
    if (unlikely(cpy_r_r87 == NULL)) {
        CPy_AddTraceback("generate_urn.py", "<module>", 64, CPyStatic_globals);
        goto CPyL47;
    } else
        goto CPyL35;
CPyL35: ;
    cpy_r_r88 = CPyModule_builtins;
    cpy_r_r89 = CPyStatic_unicode_12; /* 'print' */
    cpy_r_r90 = CPyObject_GetAttr(cpy_r_r88, cpy_r_r89);
    if (unlikely(cpy_r_r90 == NULL)) {
        CPy_AddTraceback("generate_urn.py", "<module>", 64, CPyStatic_globals);
        goto CPyL48;
    } else
        goto CPyL36;
CPyL36: ;
    cpy_r_r91 = PyObject_CallFunctionObjArgs(cpy_r_r90, cpy_r_r87, NULL);
    CPy_DecRef(cpy_r_r90);
    CPy_DecRef(cpy_r_r87);
    if (unlikely(cpy_r_r91 == NULL)) {
        CPy_AddTraceback("generate_urn.py", "<module>", 64, CPyStatic_globals);
        goto CPyL47;
    } else
        goto CPyL49;
CPyL37: ;
    cpy_r_r92 = CPyModule_time;
    cpy_r_r93 = CPyStatic_unicode_8; /* 'time' */
    cpy_r_r94 = CPyObject_GetAttr(cpy_r_r92, cpy_r_r93);
    if (unlikely(cpy_r_r94 == NULL)) {
        CPy_AddTraceback("generate_urn.py", "<module>", 65, CPyStatic_globals);
        goto CPyL47;
    } else
        goto CPyL38;
CPyL38: ;
    cpy_r_r95 = PyObject_CallFunctionObjArgs(cpy_r_r94, NULL);
    CPy_DecRef(cpy_r_r94);
    if (unlikely(cpy_r_r95 == NULL)) {
        CPy_AddTraceback("generate_urn.py", "<module>", 65, CPyStatic_globals);
        goto CPyL47;
    } else
        goto CPyL39;
CPyL39: ;
    if (likely(CPyFloat_Check(cpy_r_r95)))
        cpy_r_r96 = cpy_r_r95;
    else {
        CPy_TypeError("float", cpy_r_r95);
        cpy_r_r96 = NULL;
    }
    if (unlikely(cpy_r_r96 == NULL)) {
        CPy_AddTraceback("generate_urn.py", "<module>", 65, CPyStatic_globals);
        goto CPyL47;
    } else
        goto CPyL40;
CPyL40: ;
    cpy_r_r97 = CPyStatic_globals;
    cpy_r_r98 = CPyStatic_unicode_14; /* 't0' */
    cpy_r_r99 = CPyDict_GetItem(cpy_r_r97, cpy_r_r98);
    if (unlikely(cpy_r_r99 == NULL)) {
        CPy_AddTraceback("generate_urn.py", "<module>", 65, CPyStatic_globals);
        goto CPyL50;
    } else
        goto CPyL41;
CPyL41: ;
    if (likely(CPyFloat_Check(cpy_r_r99)))
        cpy_r_r100 = cpy_r_r99;
    else {
        CPy_TypeError("float", cpy_r_r99);
        cpy_r_r100 = NULL;
    }
    if (unlikely(cpy_r_r100 == NULL)) {
        CPy_AddTraceback("generate_urn.py", "<module>", 65, CPyStatic_globals);
        goto CPyL50;
    } else
        goto CPyL42;
CPyL42: ;
    cpy_r_r101 = PyNumber_Subtract(cpy_r_r96, cpy_r_r100);
    CPy_DecRef(cpy_r_r96);
    CPy_DecRef(cpy_r_r100);
    if (unlikely(cpy_r_r101 == NULL)) {
        CPy_AddTraceback("generate_urn.py", "<module>", 65, CPyStatic_globals);
        goto CPyL47;
    } else
        goto CPyL43;
CPyL43: ;
    if (likely(CPyFloat_Check(cpy_r_r101)))
        cpy_r_r102 = cpy_r_r101;
    else {
        CPy_TypeError("float", cpy_r_r101);
        cpy_r_r102 = NULL;
    }
    if (unlikely(cpy_r_r102 == NULL)) {
        CPy_AddTraceback("generate_urn.py", "<module>", 65, CPyStatic_globals);
        goto CPyL47;
    } else
        goto CPyL44;
CPyL44: ;
    cpy_r_r103 = CPyModule_builtins;
    cpy_r_r104 = CPyStatic_unicode_12; /* 'print' */
    cpy_r_r105 = CPyObject_GetAttr(cpy_r_r103, cpy_r_r104);
    if (unlikely(cpy_r_r105 == NULL)) {
        CPy_AddTraceback("generate_urn.py", "<module>", 65, CPyStatic_globals);
        goto CPyL51;
    } else
        goto CPyL45;
CPyL45: ;
    cpy_r_r106 = PyObject_CallFunctionObjArgs(cpy_r_r105, cpy_r_r102, NULL);
    CPy_DecRef(cpy_r_r105);
    CPy_DecRef(cpy_r_r102);
    if (unlikely(cpy_r_r106 == NULL)) {
        CPy_AddTraceback("generate_urn.py", "<module>", 65, CPyStatic_globals);
        goto CPyL47;
    } else
        goto CPyL52;
CPyL46: ;
    return 1;
CPyL47: ;
    cpy_r_r107 = 2;
    return cpy_r_r107;
CPyL48: ;
    CPy_DecRef(cpy_r_r87);
    goto CPyL47;
CPyL49: ;
    CPy_DecRef(cpy_r_r91);
    goto CPyL37;
CPyL50: ;
    CPy_DecRef(cpy_r_r96);
    goto CPyL47;
CPyL51: ;
    CPy_DecRef(cpy_r_r102);
    goto CPyL47;
CPyL52: ;
    CPy_DecRef(cpy_r_r106);
    goto CPyL46;
}

int CPyGlobalsInit(void)
{
    static int is_initialized = 0;
    if (is_initialized) return 0;
    
    CPy_Init();
    CPyModule_generate_urn = Py_None;
    CPyModule_builtins = Py_None;
    CPyModule_typing = Py_None;
    CPyModule_math = Py_None;
    CPyModule_random = Py_None;
    CPyModule_time = Py_None;
    CPyStatic_unicode_0 = PyUnicode_FromStringAndSize("builtins", 8);
    if (unlikely(CPyStatic_unicode_0 == NULL))
        return -1;
    CPyStatic_unicode_1 = PyUnicode_FromStringAndSize("typing", 6);
    if (unlikely(CPyStatic_unicode_1 == NULL))
        return -1;
    CPyStatic_unicode_2 = PyUnicode_FromStringAndSize("List", 4);
    if (unlikely(CPyStatic_unicode_2 == NULL))
        return -1;
    CPyStatic_unicode_3 = PyUnicode_FromStringAndSize("Any", 3);
    if (unlikely(CPyStatic_unicode_3 == NULL))
        return -1;
    CPyStatic_unicode_4 = PyUnicode_FromStringAndSize("Dict", 4);
    if (unlikely(CPyStatic_unicode_4 == NULL))
        return -1;
    CPyStatic_unicode_5 = PyUnicode_FromStringAndSize("Tuple", 5);
    if (unlikely(CPyStatic_unicode_5 == NULL))
        return -1;
    CPyStatic_unicode_6 = PyUnicode_FromStringAndSize("math", 4);
    if (unlikely(CPyStatic_unicode_6 == NULL))
        return -1;
    CPyStatic_unicode_7 = PyUnicode_FromStringAndSize("random", 6);
    if (unlikely(CPyStatic_unicode_7 == NULL))
        return -1;
    CPyStatic_unicode_8 = PyUnicode_FromStringAndSize("time", 4);
    if (unlikely(CPyStatic_unicode_8 == NULL))
        return -1;
    CPyStatic_unicode_9 = PyUnicode_FromStringAndSize("factorial", 9);
    if (unlikely(CPyStatic_unicode_9 == NULL))
        return -1;
    CPyStatic_unicode_10 = PyUnicode_FromStringAndSize("randint", 7);
    if (unlikely(CPyStatic_unicode_10 == NULL))
        return -1;
    CPyStatic_unicode_11 = PyUnicode_FromStringAndSize("We Have a problem... replace fell through....", 45);
    if (unlikely(CPyStatic_unicode_11 == NULL))
        return -1;
    CPyStatic_unicode_12 = PyUnicode_FromStringAndSize("print", 5);
    if (unlikely(CPyStatic_unicode_12 == NULL))
        return -1;
    CPyStatic_unicode_13 = PyUnicode_FromStringAndSize("exit", 4);
    if (unlikely(CPyStatic_unicode_13 == NULL))
        return -1;
    CPyStatic_unicode_14 = PyUnicode_FromStringAndSize("t0", 2);
    if (unlikely(CPyStatic_unicode_14 == NULL))
        return -1;
    is_initialized = 1;
    return 0;
}

PyObject *CPyStatic_unicode_0;
PyObject *CPyStatic_unicode_1;
PyObject *CPyStatic_unicode_2;
PyObject *CPyStatic_unicode_3;
PyObject *CPyStatic_unicode_4;
PyObject *CPyStatic_unicode_5;
PyObject *CPyStatic_unicode_6;
PyObject *CPyStatic_unicode_7;
PyObject *CPyStatic_unicode_8;
PyObject *CPyStatic_unicode_9;
PyObject *CPyStatic_unicode_10;
PyObject *CPyStatic_unicode_11;
PyObject *CPyStatic_unicode_12;
PyObject *CPyStatic_unicode_13;
PyObject *CPyStatic_unicode_14;
CPyModule *CPyModule_generate_urn_internal = NULL;
CPyModule *CPyModule_generate_urn;
PyObject *CPyStatic_globals;
CPyModule *CPyModule_builtins_internal = NULL;
CPyModule *CPyModule_builtins;
CPyModule *CPyModule_typing_internal = NULL;
CPyModule *CPyModule_typing;
CPyModule *CPyModule_math_internal = NULL;
CPyModule *CPyModule_math;
CPyModule *CPyModule_random_internal = NULL;
CPyModule *CPyModule_random;
CPyModule *CPyModule_time_internal = NULL;
CPyModule *CPyModule_time;
PyObject *CPyDef_gen_urn(CPyTagged cpy_r_numvotes, CPyTagged cpy_r_replace, PyObject *cpy_r_alts);
PyObject *CPyPy_gen_urn(PyObject *self, PyObject *args, PyObject *kw);
PyObject *CPyDef_gen_ic_vote(PyObject *cpy_r_alts);
PyObject *CPyPy_gen_ic_vote(PyObject *self, PyObject *args, PyObject *kw);
char CPyDef___top_level__(void);
