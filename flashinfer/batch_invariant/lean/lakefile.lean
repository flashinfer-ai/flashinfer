import Lake
open Lake DSL

-- The kernel decides the generated conformance theorems by evaluating `String` equalities of several KB (a list of
-- characters: the recursion is as deep as the string); the default 8 MB thread stack overflows near 6.4 KB, so the
-- build runs `lean` with a 64 MB thread stack (an argument outside the trace: it changes no build result).
package bikernels where
  weakLeanArgs := #["--tstack=65536"]

@[default_target]
lean_lib BIKernels

lean_exe conformance where
  root := `Main

lean_exe quantconformance where
  root := `QuantMain

lean_exe samplingconformance where
  root := `SamplingMain

lean_exe megamoeconformance where
  root := `MegaMoEMain

lean_exe nvfp4megamoeconformance where
  root := `NVFP4MegaMoEMain

lean_exe mxfp4megamoeconformance where
  root := `MXFP4MegaMoEMain

lean_exe mxfp8megamoeconformance where
  root := `MXFP8MegaMoEMain

lean_exe axioms where
  root := `Axioms
  supportInterpreter := true

lean_exe moeconformance where
  root := `MoEMain

lean_exe numericalmegamoeconformance where
  root := `NumericalMegaMoEMain

lean_exe a16megamoeconformance where
  root := `A16MegaMoEMain
