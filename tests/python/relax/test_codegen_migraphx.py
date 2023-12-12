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
import numpy as np
import pytest

import tvm
import tvm.testing
import tvm.topi.testing
from tvm import relax
from tvm import dlight as dl
from tvm.relax.backend.contrib.migraphx import partition_for_migraphx
from tvm.relax.testing import get_relax_matmul_module
from tvm.script import relax as R


@pytest.fixture(autouse=True)
def reset_seed():
    np.random.seed(0)


has_migraphx = tvm.get_global_func("relax.is_migraphx_runtime_enabled")()

migraphx_enabled = pytest.mark.skipif(
    not has_migraphx,
    reason="CUBLAS not enabled.",
)

pytestmark = [migraphx_enabled]


def build_and_run(mod, inputs_np, target):
    dev = tvm.device(target, 0)
    with tvm.transform.PassContext(
        config={
        }
    ):
        ex = relax.build(mod, target)
    vm = relax.VirtualMachine(ex, dev)
    f = vm["main"]
    inputs = [tvm.nd.array(inp, dev) for inp in inputs_np]

    return f(*inputs).numpy()


def get_result_with_relax_migraphx_offload(mod, np_inputs):
    mod = partition_for_migraphx(mod)
    mod = relax.transform.RunCodegen()(mod)
    # with tvm.target.Target("rocm"):
    #     # mod = rx.get_pipeline("zero")(mod)  # pylint: disable=no-value-for-parameter
    #     mod = dl.ApplyDefaultSchedule(  # pylint: disable=not-callable
    #         dl.gpu.Matmul(),
    #         dl.gpu.GEMV(),
    #         dl.gpu.Reduction(),
    #         dl.gpu.GeneralReduction(),
    #         dl.gpu.Transpose(),
    #         dl.gpu.Fallback(),
    #     )(mod)
    with tvm.target.Target("rocm"):
        mod = tvm.tir.transform.DefaultGPUSchedule()(mod)
    print(mod)

    return build_and_run(mod, np_inputs, "rocm")


def _to_concrete_shape(symbolic_shape, var_table):
    result = []
    for dim in symbolic_shape:
        if not isinstance(dim, tvm.tir.expr.Var):
            result.append(dim)
            continue

        if dim not in var_table:
            var_table[dim] = np.random.randint(10, 50)
        result.append(var_table[dim])

    return tuple(result)


_vars = {
    "a": tvm.tir.expr.Var("a", "int64"),
    "b": tvm.tir.expr.Var("b", "int64"),
}


_epilogue_table = {
    "none": (False, None),
    "bias": (True, None),
    "relu": (True, R.nn.relu),
    "gelu": (True, R.nn.gelu),
}


@pytest.mark.parametrize(
    "x_shape, y_shape, transpose_y, epilogue",
    [
        # Regular
        # ((8, 8), (8, 8), False, "none"),
        # ((8, 16), (16, 8), False, "none"),
        ((1, 1, 4096), (4096, 27392), False, "none"),
        # ((_vars["a"], 6), (6, 16), False, "none"),
        # ((_vars["a"], 6), (6, 16), False, "bias"),
        # Transposed
        # ((4, 16), (16, 128), True, "relu"),
        # ((35, 8), (8, 8), True, "gelu"),
        # # 3D x 3D
        # ((6, 32, 8), (6, 8, 10), False, "none"),
        # ((6, 32, 8), (6, 8, 10), True, "none"),
        # ((_vars["a"], 32, 8), (_vars["a"], 8, 10), True, "gelu"),
        # ND x ND
        # ((5, 3, 32, 8), (5, 3, 8, 10), False, "none"),
        # ND x 2D
        # ((5, 3, 32, 8), (8, 10), False, "none"),
    ],
)
@pytest.mark.parametrize(
    "in_dtype, out_dtype",
    [
        # ("float16", "float16"),
        ("float32", "float32"),
    ],
)
def test_matmul_offload(
    x_shape,
    y_shape,
    transpose_y,
    epilogue,
    in_dtype,
    out_dtype,
):
    with_bias, activation = _epilogue_table[epilogue]
    var_table = {}
    concrete_x_shape = _to_concrete_shape(x_shape, var_table)
    concrete_y_shape = _to_concrete_shape(y_shape, var_table)
    x = np.random.randn(*concrete_x_shape).astype(in_dtype)
    y = np.random.randn(*concrete_y_shape).astype(in_dtype)

    if transpose_y:
        y = np.swapaxes(y, -2, -1)
        y_shape = (*y_shape[:-2], y_shape[-1], y_shape[-2])

    if with_bias:
        bias = np.random.randn(concrete_y_shape[-1]).astype(out_dtype)
        args = (x, y, bias)
    else:
        bias = None
        args = (x, y)

    mod = get_relax_matmul_module(
        x_shape,
        y_shape,
        in_dtype,
        out_dtype,
        bias_shape=bias.shape if with_bias else None,
        transposed_y=transpose_y,
        activation=activation,
    )

    out = get_result_with_relax_migraphx_offload(mod, args)
    ref = build_and_run(mod, args, "llvm")

    tvm.testing.assert_allclose(out, ref, rtol=1e-2, atol=1e-2)

def test_matmul():
    @tvm.script.ir.ir_module
    class Mod:
        @R.function
        def main(
            x: R.Tensor((16, 16), "float16"),
            w0: R.Tensor((16, 16), "float16"),
            w1: R.Tensor((16, 16), "float16"),
            w2: R.Tensor((16, 16), "float16"),
        ):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv0 = R.matmul(x, w0)
                lv1 = R.matmul(lv0, w1)
                lv2 = R.matmul(lv1, w2)
                R.output(lv2)
            return lv2

    mod = Mod
    shape = [16, 16]
    data = np.random.rand(*shape).astype(np.float16)
    w0 = np.random.rand(*shape).astype(np.float16)
    w1 = np.random.rand(*shape).astype(np.float16)
    w2 = np.random.rand(*shape).astype(np.float16)
    inputs = (data, w0, w1, w2)

    out = get_result_with_relax_migraphx_offload(Mod, inputs)
    ref = build_and_run(Mod, inputs, "llvm")

    tvm.testing.assert_allclose(out, ref, rtol=1e-2, atol=1e-2)

def test_transposed_matmul():
    @tvm.script.ir_module
    class TransposedMatmul:
        @R.function
        def main(
            x: R.Tensor((4, 4), "float32"),
            y: R.Tensor((4, 4), "float32"),
        ):
            with R.dataflow():
                lv1 = R.permute_dims(y, [1, 0])
                # Because lv1 is used by both lv2 and out, it should stay out of
                # the fused function. Otherwise the fused function will return
                # tuple output, which isn't possible in cutlass, e.g.
                # @R.function
                # def fused_relax_permute_dims_relax_matmul(...):
                #     R.func_attr({"Composite": "cutlass.matmul_transposed", "Primitive": 1})
                #     with R.dataflow():
                #         gv: R.Tensor((4, 4), dtype="float32") = R.permute_dims(y, axes=None)
                #         gv1: R.Tensor((4, 4), dtype="float32") = R.matmul(x, gv, out_dtype="void")
                #         R.output(gv, gv1)
                #     return (gv, gv1)  # Cannot get `gv` if dispatch to cutlass kernel.
                lv2 = R.matmul(x, lv1)
                out = R.matmul(lv1, lv2)
                R.output(out)

            return out

    x = np.random.randn(4, 4).astype("float32")
    y = np.random.randn(4, 4).astype("float32")
    inputs = (x, y)
    out = get_result_with_relax_migraphx_offload(TransposedMatmul, inputs)
    ref = build_and_run(TransposedMatmul, inputs, "llvm")

    tvm.testing.assert_allclose(out, ref, rtol=1e-2, atol=1e-2)

def matmul_offload1(
    x_shape=(8,8),
    y_shape=(8,8),
    transpose_y=False,
    epilogue="none",
    in_dtype="float32",
    out_dtype="float32",
):
    with_bias, activation = _epilogue_table[epilogue]
    var_table = {}
    concrete_x_shape = _to_concrete_shape(x_shape, var_table)
    concrete_y_shape = _to_concrete_shape(y_shape, var_table)
    x = np.random.randn(*concrete_x_shape).astype(in_dtype)
    y = np.random.randn(*concrete_y_shape).astype(in_dtype)

    if transpose_y:
        y = np.swapaxes(y, -2, -1)
        y_shape = (*y_shape[:-2], y_shape[-1], y_shape[-2])

    if with_bias:
        bias = np.random.randn(concrete_y_shape[-1]).astype(out_dtype)
        args = (x, y, bias)
    else:
        bias = None
        args = (x, y)

    mod = get_relax_matmul_module(
        x_shape,
        y_shape,
        in_dtype,
        out_dtype,
        bias_shape=bias.shape if with_bias else None,
        transposed_y=transpose_y,
        activation=activation,
    )

    out = get_result_with_relax_migraphx_offload(mod, args)
    ref = build_and_run(mod, args, "llvm")

    print(out)
    print(ref)

    tvm.testing.assert_allclose(out, ref, rtol=1e-2, atol=1e-2)


def test_migraphx_partition_matmul_without_bias():
    # cuBLAS does not handle 2D bias (residual input)
    mod = get_relax_matmul_module((16, 32), (32, 32), "float16", "float16", bias_shape=(16, 32))
    mod = partition_for_migraphx(mod)

    # R.add is still in the main function
    assert len(mod["main"].body.blocks[0].bindings) == 2

if __name__ == "__main__":
    # tvm.testing.main()
    # test_transposed_matmul()
    test_matmul()
    # test_matmul_offload1((1, 1, 4096), (4096, 27392), False, "none", "float16", "float16")
    # test_matmul_offload1(( 1, 4096), (4096, 27392), False, "none", "float16", "float16")
    # test_matmul_offload1((1, 1, 4096), (4096, 27392), False, "none", "float32", "float32")
    # test_matmul_offload1((128, 128), (128, 128), False, "none", "float32", "float32")
    # test_matmul_offload1((128, 24), (24, 128), False, "none", "float16", "float16")
    # test_matmul_offload1((_vars["a"],16), (16, 8))
