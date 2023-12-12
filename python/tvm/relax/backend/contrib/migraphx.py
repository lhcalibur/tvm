# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name, unused-argument, import-outside-toplevel
"""Pattern table and codegen for CoreML"""

import os
import shutil
import tvm._ffi
from tvm.contrib import coreml_runtime
from tvm.contrib.xcode import compile_coreml

import tvm
from tvm.relax import transform
from tvm.relax.struct_info import TensorStructInfo, PrimStructInfo
from tvm.relax.expr import (
    BindingBlock,
    Call,
    Function,
    PrimValue,
    SeqExpr,
    Var,
    VarBinding,
    Constant,
)
from tvm.relax.dpl.pattern import is_op, wildcard
from tvm.relax.transform import PatternCheckContext
from ..pattern_registry import get_patterns_with_prefix, register_patterns
from ..patterns import make_matmul_pattern
from ...expr_functor import PyExprVisitor, visitor
from ..utils import has_leaking_intermediate_variables


def _check_default(context: PatternCheckContext) -> bool:
    return True


def _is_supported_dtype(lhs_dtype, rhs_dtype):
    """Check if dtypes in the given workload are supported by cuBLAS BYOC."""
    return (
        (lhs_dtype == "float16" and rhs_dtype == "float16")
        or (lhs_dtype == "float32" and rhs_dtype == "float32")
        or (lhs_dtype == "int8" and rhs_dtype == "int8")
    )


def _check_matmul(context: PatternCheckContext) -> bool:
    if has_leaking_intermediate_variables(context):
        return False
    lhs = context.annotated_expr["lhs"]
    rhs = context.annotated_expr["rhs"]

    lhs_dtype = lhs.struct_info.dtype
    rhs_dtype = rhs.struct_info.dtype
    if not _is_supported_dtype(lhs_dtype, rhs_dtype):
        return False

    lhs_shape = lhs.struct_info.shape.values
    rhs_shape = rhs.struct_info.shape.values

    for shape in [lhs_shape, rhs_shape]:
        for s in shape:
            if not isinstance(s, (tvm.tir.expr.IntImm, int)):
                # Must be constant
                return False

    return True


def default_binary_patterns(op_name: str):
    """
    Returns a list of binary op patterns in coreML BYOC backend.
    """

    def _make_binary_pattern():
        lhs = wildcard()
        rhs = wildcard()
        out = is_op("relax." + op_name)(lhs, rhs)
        annotations = {"lhs": lhs, "rhs": rhs, "root": out}
        return out, annotations

    def _binary_pattern(pattern_name):
        return (pattern_name, *_make_binary_pattern(), _check_default)

    return [_binary_pattern("migraphx." + op_name)]


def default_unary_patterns(op_name: str):
    """
    Returns a list of unary op patterns in coreML BYOC backend.
    """

    def _make_unary_pattern():
        lhs = wildcard()
        out = is_op("relax." + op_name)(lhs)
        annotations = {"lhs": lhs, "root": out}
        return out, annotations

    def _unary_pattern(pattern_name):
        return (pattern_name, *_make_unary_pattern(), _check_default)

    return [_unary_pattern("migraphx." + op_name)]


def conv2d_patterns():
    """
    Returns a list of conv2d patterns in coreML BYOC backend.
    """

    def _make_conv2d_pattern():
        lhs = wildcard()
        rhs = wildcard()
        out = is_op("relax.nn.conv2d")(lhs, rhs)
        annotations = {"lhs": lhs, "rhs": rhs, "root": out}
        return out, annotations

    def _conv2d_pattern(pattern_name):
        return (pattern_name, *_make_conv2d_pattern(), _check_default)

    return [_conv2d_pattern("migraphx.nn.conv2d")]


def matmul_patterns():
    """
    Returns a list of all matmul patterns in coreML BYOC backend.
    """

    def _matmul_pattern(pattern_name):
        return (
            pattern_name,
            *make_matmul_pattern(with_bias=False),
            _check_matmul,
            # _check_default,
        )

    return [_matmul_pattern("migraphx.matmul")]


def clip_patterns():
    """
    Returns a list of clip patterns in coreML BYOC backend.
    """

    def _make_clip_pattern():
        arg0 = wildcard()
        arg1 = wildcard()
        arg2 = wildcard()
        out = is_op("relax.clip")(arg0, arg1, arg2)
        annotations = {"arg0": arg0, "arg1": arg1, "arg2": arg2, "root": out}
        return out, annotations

    def _clip_pattern(pattern_name):
        return (pattern_name, *_make_clip_pattern(), _check_default)

    return [_clip_pattern("migraphx.clip")]


register_patterns(
    [
        *matmul_patterns(),
    ]
)


def partition_for_migraphx(mod):
    """
    Partition the input module into coreml-supported subgraphs.

    Parameters
    ----------
    mod: tvm.IRModule
        The IRModule to be partitioned.

    Returns
    -------
    mod: tvm.IRModule
        The resulting IRModule, containing partitioned subgraphs to be
        offloaded to the coreml backend.
    """

    patterns = get_patterns_with_prefix("migraphx")
    mod = transform.FuseOpsByPattern(patterns, bind_constants=True, annotate_codegen=False)(mod)
    mod = transform.MergeCompositeFunctions()(mod)
    return mod


@visitor
class CallNodeInfoCollector(PyExprVisitor):
    """
    Collect PrimValue, Constant and attributes in the inner function
    """

    def __init__(self, op_name):
        self.primvals = []
        self.attrs = []
        self.consts = []
        self.op_name = op_name

    def visit_call_(self, call: Call) -> None:
        self.attrs.append(call.attrs)
        for arg in call.args:
            if isinstance(arg, PrimValue):
                self.primvals.append(arg)
            if isinstance(arg, Constant):
                self.consts.append(arg)

    def collect(self, expr):
        self.visit_expr(expr)
        return self.primvals, self.attrs, self.consts
