from typing import List, Optional

from .op import Op
from .types import Sort
from .validators import CompileError, ValidityCheck, get_default_validity_checks
from .fusion_rules import FusionRule, get_default_fusion_rules


class Compiler:
    def __init__(self):
        self.fusion_rules: List[FusionRule] = []
        self.validity_checks: List[ValidityCheck] = []
        self._install_defaults()
        
    def register_fusion_rule(self, rule: FusionRule) -> None:
        self.fusion_rules.append(rule)
        self.fusion_rules.sort(key=lambda r: -r.prio)
    
    def register_validity_check(self, check: ValidityCheck) -> None:
        self.validity_checks.append(check)
    
    def compile(self, ops: List[Op]) -> List[Op]:
        if not ops:
            raise CompileError("Cannot compile empty operator list")
        
        compiled_ops = list(ops)
        
        self._type_check(compiled_ops)

        self._run_validity_checks(compiled_ops)
         
        compiled_ops = self._fuse_all(compiled_ops)
        
        return compiled_ops
        
    def _type_check(self, ops: List[Op]) -> None:
        current_sort = Sort.LOGITS
        
        for i, op in enumerate(ops):
            compatible_found = False
            
            for j, input_sort in enumerate(op.IN):
                if current_sort == input_sort:
                    current_sort = op.OUT[j]
                    compatible_found = True
                    break
            
            if not compatible_found:
                raise CompileError(
                    f"Type mismatch at operator {i} ({op.__class__.__name__}). "
                    f"Expected input sort {current_sort}, but operator accepts {op.IN}"
                )
            
    
    def _run_validity_checks(self, ops: List[Op]) -> None:
        for check in self.validity_checks:
            check(ops)
    
    def _fuse_all(self, ops: List[Op]) -> List[Op]:
        i = 0
        while i < len(ops):
            fusion_applied = False
            
            for rule in self.fusion_rules:
                span = len(rule.pattern)
                
                if i + span > len(ops):
                    continue
                
                window = ops[i:i + span]
                
                if (self._pattern_matches(window, rule.pattern) and 
                    rule.guard(window)):
                    
                    fused_op = rule.build(window)
                    ops[i:i + span] = [fused_op]
                    
                    i = max(i - 1, 0)
                    fusion_applied = True
                    break
            
            if not fusion_applied:
                i += 1
        
        return ops
    
    def _pattern_matches(self, window: List[Op], pattern: tuple) -> bool:
        if len(window) != len(pattern):
            return False
        
        return all(isinstance(window[i], pattern[i]) for i in range(len(pattern)))
    
    
    def _install_defaults(self) -> None:
        for check in get_default_validity_checks():
            self.validity_checks.append(check)

        for rule in get_default_fusion_rules():
            self.register_fusion_rule(rule)


def compile_pipeline(ops: List[Op], 
                    custom_fusion_rules: Optional[List[FusionRule]] = None,
                    custom_validity_checks: Optional[List[ValidityCheck]] = None) -> List[Op]:
    compiler = Compiler()
    
    if custom_fusion_rules:
        for rule in custom_fusion_rules:
            compiler.register_fusion_rule(rule)
    
    if custom_validity_checks:
        for check in custom_validity_checks:
            compiler.register_validity_check(check)
    
    return compiler.compile(ops) 