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

from ..array_utils import as_same_device
from ..operands import TensorFuse
from .core import TensorFuseChunkMixin


class TensorJAXFuseChunk(TensorFuse, TensorFuseChunkMixin):
    _op_type_ = None  # no opcode, cannot be serialized

    @classmethod
    def execute(cls, ctx, op):
        chunk = op.outputs[0]
        inputs = as_same_device([ctx[c.key] for c in op.inputs], device=op.device)
        functions = [_get_jax_function(operand) for operand in op.operands]
        jax_func = jax.jit(_fusion)
        ctx[chunk.key] = np.asarray(jax_func(inputs, functions))


def _fusion(inputs, functions):
    inputs = functions[0](*inputs)
    for func in functions[1:]:
        inputs = func(inputs)
    return inputs


def _get_jax_function(operand):
    from functools import partial

    import jax.numpy as jnp

    func = getattr(jnp, getattr(operand, "_func_name"))

    if len(operand.inputs) == 1:
        if np.isscalar(operand.lhs):
            left = operand.lhs
            return partial(func, left)
        if np.isscalar(operand.rhs):
            right = operand.rhs
            return lambda x: func(x, right)
    else:
        return func
