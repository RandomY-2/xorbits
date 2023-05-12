# # Copyright 2022-2023 XProbe Inc.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #      http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.

# import dataclasses
# import functools
# import logging
# from typing import List, Set

# import jax.numpy as jnp
# import numpy as np

# from ...core import ChunkGraph, ChunkType
# from ...tensor import arithmetic, reduction
# from ...tensor.fuse import TensorJAXFuseChunk
# from ...tensor.fuse.jax import JAX_INSTALLED
# from .core import RuntimeOptimizer, register_optimizer

# logger = logging.getLogger(__name__)


# @dataclasses.dataclass
# class _Fuse:
#     graph: ChunkGraph
#     heads: List[ChunkType]
#     tails: List[ChunkType]


# def _can_fuse(node: ChunkType):
#     op = node.op
#     op_type = type(op)
#     return hasattr(jnp, op_type["_func_name"])
