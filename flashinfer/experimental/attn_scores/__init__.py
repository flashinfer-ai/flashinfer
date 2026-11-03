# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Experimental MQA TopK backends; imported only on explicit use."""

import functools
import threading


@functools.cache
def _wrapper_type():
    """Load definitions without exposing constructor tensor owners to DSL frames.

    CuTe DSL 4.7 decorators can retain their importing call stack until cyclic
    collection. A definition-only thread has no constructor arguments; loading
    finishes before a wrapper or its CUDA storage exists. Compilation is lazy.
    """
    outcome = []

    def load():
        try:
            from .paged_delegate import FP8PagedMQATopKWrapper
        except BaseException as error:
            outcome.append(error)
        else:
            outcome.append(FP8PagedMQATopKWrapper)

    thread = threading.Thread(target=load, name="flashinfer-topk-definitions")
    thread.start()
    thread.join()
    result = outcome[0]
    if isinstance(result, BaseException):
        # Preserve the dependency's type, message and importing traceback.
        raise result
    return result


@functools.cache
def _continuous_plan_type(backend):
    outcome = []

    def load():
        try:
            # Load the shared DSL definitions without retaining caller tensors.
            from . import selective_logits  # noqa: F401

            if backend == "v2":
                from .selective_v2 import _CapacityV2TopKPlan as Plan
            elif backend == "full":
                from .full_contiguous import _ContinuousFullTopKPlan as Plan
            else:
                from .selective_contiguous import _ContinuousSelectiveTopKPlan as Plan
            outcome.append(Plan)
        except BaseException as error:
            outcome.append(error)

    thread = threading.Thread(
        target=load, name="flashinfer-continuous-topk-definitions"
    )
    thread.start()
    thread.join()
    if isinstance(outcome[0], BaseException):
        raise outcome[0]
    return outcome[0]
