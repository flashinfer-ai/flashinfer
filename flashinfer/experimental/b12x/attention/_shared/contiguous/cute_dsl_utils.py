from dataclasses import dataclass, fields

import cutlass
import cutlass.cute as cute
from cutlass.base_dsl.typing import JitArgument
from cutlass.cutlass_dsl import NumericMeta

StaticTypes = (cutlass.Constexpr, NumericMeta, int, bool, str, float, type(None))


def _partition_fields(obj):
    all_fields = {field.name: getattr(obj, field.name) for field in fields(obj)}
    constexpr = {
        name: value
        for name, value in all_fields.items()
        if isinstance(value, StaticTypes)
    }
    non_constexpr = {
        name: value
        for name, value in all_fields.items()
        if not isinstance(value, StaticTypes)
    }
    return constexpr, non_constexpr


def _new_from_mlir_values(self, values):
    constexpr_fields, non_constexpr_fields = _partition_fields(self)
    for (name, field), n_items in zip(non_constexpr_fields.items(), self._values_pos):
        non_constexpr_fields[name] = cutlass.new_from_mlir_values(
            field, values[:n_items]
        )
        values = values[n_items:]
    return self.__class__(**non_constexpr_fields, **constexpr_fields)


@dataclass
class ParamsBase(JitArgument):
    def __c_pointers__(self):
        all_fields = [getattr(self, field.name) for field in fields(self)]
        non_constexpr_fields = [
            field for field in all_fields if not isinstance(field, StaticTypes)
        ]
        c_ptrs = []
        for obj in non_constexpr_fields:
            if hasattr(obj, "__c_pointers__"):
                c_ptrs.extend(obj.__c_pointers__())
        return c_ptrs

    def __get_mlir_types__(self):
        all_fields = [getattr(self, field.name) for field in fields(self)]
        non_constexpr_fields = [
            field for field in all_fields if not isinstance(field, StaticTypes)
        ]
        types, self._values_pos = [], []
        for obj in non_constexpr_fields:
            if hasattr(obj, "__get_mlir_types__"):
                obj_types = obj.__get_mlir_types__()
            else:
                obj_values = cutlass.extract_mlir_values(obj)
                obj_types = [value.type for value in obj_values]
            types.extend(obj_types)
            self._values_pos.append(len(obj_types))
        return types

    def __extract_mlir_values__(self):
        _, non_constexpr_fields = _partition_fields(self)
        values, self._values_pos = [], []
        for obj in non_constexpr_fields.values():
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    __new_from_mlir_values__ = _new_from_mlir_values


def assume_strides_aligned(t):
    divby = 128 // t.element_type.width
    strides = tuple(
        s if isinstance(s, int) else cute.assume(s, divby=divby) for s in t.stride[:-1]
    )
    return (*strides, t.stride[-1])


def assume_tensor_aligned(t):
    if t is None:
        return None
    return cute.make_tensor(
        t.iterator, cute.make_layout(t.shape, stride=assume_strides_aligned(t))
    )
