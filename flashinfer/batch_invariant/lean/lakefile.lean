import Lake
open Lake DSL

package bikernels

@[default_target]
lean_lib BIKernels

lean_exe conformance where
  root := `Main

lean_exe quantconformance where
  root := `QuantMain

lean_exe samplingconformance where
  root := `SamplingMain

lean_exe axioms where
  root := `Axioms
  supportInterpreter := true
