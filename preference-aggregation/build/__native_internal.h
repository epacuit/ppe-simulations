#ifndef MYPYC_NATIVE_INTERNAL_H
#define MYPYC_NATIVE_INTERNAL_H
#include <Python.h>
#include <CPy.h>
#include "__native.h"

int CPyGlobalsInit(void);

extern PyObject *CPyStatic_unicode_0;
extern PyObject *CPyStatic_unicode_1;
extern PyObject *CPyStatic_unicode_2;
extern PyObject *CPyStatic_unicode_3;
extern PyObject *CPyStatic_unicode_4;
extern PyObject *CPyStatic_unicode_5;
extern PyObject *CPyStatic_unicode_6;
extern PyObject *CPyStatic_unicode_7;
extern PyObject *CPyStatic_unicode_8;
extern PyObject *CPyStatic_unicode_9;
extern PyObject *CPyStatic_unicode_10;
extern PyObject *CPyStatic_unicode_11;
extern PyObject *CPyStatic_unicode_12;
extern PyObject *CPyStatic_unicode_13;
extern PyObject *CPyStatic_unicode_14;
extern CPyModule *CPyModule_generate_urn_internal;
extern CPyModule *CPyModule_generate_urn;
extern PyObject *CPyStatic_globals;
extern CPyModule *CPyModule_builtins_internal;
extern CPyModule *CPyModule_builtins;
extern CPyModule *CPyModule_typing_internal;
extern CPyModule *CPyModule_typing;
extern CPyModule *CPyModule_math_internal;
extern CPyModule *CPyModule_math;
extern CPyModule *CPyModule_random_internal;
extern CPyModule *CPyModule_random;
extern CPyModule *CPyModule_time_internal;
extern CPyModule *CPyModule_time;
extern PyObject *CPyDef_gen_urn(CPyTagged cpy_r_numvotes, CPyTagged cpy_r_replace, PyObject *cpy_r_alts);
extern PyObject *CPyPy_gen_urn(PyObject *self, PyObject *args, PyObject *kw);
extern PyObject *CPyDef_gen_ic_vote(PyObject *cpy_r_alts);
extern PyObject *CPyPy_gen_ic_vote(PyObject *self, PyObject *args, PyObject *kw);
extern char CPyDef___top_level__(void);
#endif
