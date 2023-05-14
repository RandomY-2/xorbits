# Copyright 2022-2023 XProbe Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

try:
    import jax

    JAX_INSTALLED = True
except ImportError:
    JAX_INSTALLED = False
import numpy as np

from .. import arithmetic
from ..array_utils import as_same_device
from ..operands import TensorFuse
from .core import TensorFuseChunkMixin


class TensorJAXFuseChunk(TensorFuse, TensorFuseChunkMixin):
    _op_type_ = None  # no opcode, cannot be serialized

    @classmethod
    def execute(cls, ctx, op):
        chunk = op.outputs[0]
        inputs = as_same_device([ctx[c.key] for c in op.inputs], device=op.device)
        jit_func = _evaluate(chunk)
        try:
            res = jit_func(inputs)
        except Exception as e:
            raise RuntimeError(
                f"Failed to evaluate jax function {repr(jit_func)}."
            ) from e
        ctx[chunk.key] = res


def _evaluate(chunk):
    op_type = type(chunk.op)

    if op_type is TensorJAXFuseChunk:
        funcs = []
        for node in chunk.composed:
            _func = _evaluate(node)
            funcs.append(_func)

        def _fusion(inputs):
            output = funcs[0](inputs)
            for func in funcs[1:]:
                output = func(output)
            return output

        return jax.jit(_fusion)
    elif op_type in ARITHMETIC_SUPPORT:
        return _get_jax_function(chunk.op)
    else:
        raise TypeError(f"unsupported operator in jax: {op_type.__name__}")


def _get_jax_function(operand):
    from functools import partial

    import jax.numpy as jnp

    func = getattr(jnp, getattr(operand, "_func_name"))

    if len(operand.inputs) == 1 and hasattr(operand, "lhs"):
        if np.isscalar(operand.lhs):
            left = operand.lhs
            return partial(func, left)
        if np.isscalar(operand.rhs):
            right = operand.rhs
            return lambda x: func(x, right)
    else:
        return func


ARITHMETIC_SUPPORT = {
    arithmetic.TensorAdd,
    arithmetic.TensorSubtract,
    arithmetic.TensorMultiply,
    arithmetic.TensorDivide,
    arithmetic.TensorPower,
    arithmetic.TensorMod,
    arithmetic.TensorNegative,
    arithmetic.TensorAbs,
    arithmetic.TensorConj,
    arithmetic.TensorExp,
    arithmetic.TensorLog,
    arithmetic.TensorLog10,
    arithmetic.TensorExpm1,
    arithmetic.TensorLog1p,
    arithmetic.TensorSqrt,
    arithmetic.TensorEqual,
    arithmetic.TensorSin,
    arithmetic.TensorCos,
    arithmetic.TensorTan,
    arithmetic.TensorArcsin,
    arithmetic.TensorArccos,
    arithmetic.TensorArctan,
    arithmetic.TensorSinh,
    arithmetic.TensorCosh,
    arithmetic.TensorTanh,
    arithmetic.TensorArcsinh,
    arithmetic.TensorArccosh,
    arithmetic.TensorArctanh,
    arithmetic.TensorLshift,
    arithmetic.TensorRshift,
    arithmetic.TensorTreeAdd,
    arithmetic.TensorTreeMultiply,
    arithmetic.TensorFloor,
    arithmetic.TensorCeil,
}
