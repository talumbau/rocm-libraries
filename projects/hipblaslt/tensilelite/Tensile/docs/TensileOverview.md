# Tensile Package Overview

## What This Package Does

`Tensile` is a Python-based auto-tuning and kernel generation framework for GPU tensor contraction
operations (primarily GEMM — General Matrix Multiply). It generates highly optimized GPU kernels
in AMD GCN assembly, benchmarks them across a range of problem sizes, and produces a library of
hand-selected optimal solutions for later use by hipBLASLt.

The package operates in three main phases, each producing a directory of artifacts:

1. **BenchmarkProblems** — generates GPU kernels from a parameter space, compiles them, and
   benchmarks them on hardware to measure GFlops/s for various problem sizes.
2. **LibraryLogic** — analyzes the benchmark data to select the fastest solution per problem
   size, and writes YAML "logic files" describing the selection rules.
3. **LibraryClient** — generates a callable C++ library wrapping the selected kernels.

The whole workflow is driven by a YAML configuration file and invoked via the `Tensile` entry
point. The configuration describes the problem type (data types, transpose flags, contraction
indices) and the kernel parameter space to search.

---

## Key Abstractions

### Problem Representation

A **ProblemType** describes the mathematical structure of a contraction: which indices are free
(iterate over rows/cols of the output), which are batch dimensions, and which are bound
(contracted). The abstract definition lives in `Contractions.py`, and a more concrete HIP-kernel-
oriented form lives in `SolutionStructs/Problem.py`.

### Solution

A **Solution** is a dictionary of tuning parameters for a single kernel: tile sizes (`MacroTile0`,
`MacroTile1`), loop unroll depth, matrix-instruction type (MFMA/WMMA), pipeline depth, prefetch
strategy, and hundreds more. Solutions are enumerated during benchmarking by taking the Cartesian
product of parameter ranges specified in the configuration file.

### Component System

The **Component** system (`Component.py`) is a hardware-capability-aware plugin architecture.
Pieces of assembly code (MAC instructions, load/store patterns, etc.) are represented as classes
inheriting from a common base. At runtime Tensile selects the best matching implementation based
on ISA capability flags and kernel options. See `ComponentSystem.md` for details.

### KernelWriter Hierarchy

`KernelWriterBase` is the abstract base class for all kernel code generators. The main subclass
is `KernelWriter` (in `KernelWriter.py`), which uses `KernelWriterAssembly` to emit GCN assembly
for the GEMM main loop. Separate writer classes handle special-purpose kernels (beta scaling,
type conversion, reductions, activation). See `KernelWriterSystem.md` for details.

### Solution Library

A **SolutionLibrary** is a runtime data structure that maps a problem (sizes, types) to the best
solution index. Several library types exist: `SingleSolutionLibrary` (always returns one
solution), `MatchingLibrary` (nearest-neighbor lookup in a pre-built table), and others defined
in `SolutionLibrary.py`.

### Toolchain

The **Toolchain** subpackage (`Toolchain/`) wraps ROCm command-line tools — the clang assembler,
the HIP compiler, and the offload bundler — in Python objects that Tensile drives to compile
generated assembly and C++ into `.co` (code object) files. See the `Toolchain/docs/` directory
for details.

---

## Source File Map

| File | Concept |
|------|---------|
| `Tensile.py` | Entry point; `executeStepsInConfig()` orchestrates all three phases |
| `__init__.py` | Package version (`5.0.0`), root path constants |
| `BenchmarkProblems.py` | Phase 1: kernel generation, compilation, and benchmarking |
| `BenchmarkStructs.py` | `BenchmarkProcess` / `constructForkPermutations` — parameter enumeration |
| `BenchmarkSplitter.py` | Splits benchmark work across processes or cluster nodes |
| `LibraryLogic.py` | Phase 2: benchmark analysis, solution selection, logic file generation |
| `LibraryIO.py` | YAML/JSON read and write for solution files and logic files |
| `SolutionLibrary.py` | In-memory solution library types (Single, Matching, Placeholder, …) |
| `SolutionSelectionLibrary.py` | CSV-based per-problem-size solution selection helpers |
| `Contractions.py` | Abstract problem type: `FreeIndex`, `BatchIndex`, `BoundIndex`, `ProblemType` |
| `Properties.py` | Property predicates used to filter solutions in library lookups |
| `Hardware.py` | ISA / hardware capability representation |
| `Component.py` | Plugin component base class; `PartialMatch`, `Component.find()` |
| `KernelWriterBase.py` | Abstract base class `KernelWriterBase` |
| `KernelWriter.py` | Main assembly kernel writer; `ConstValues`, `MatrixInfo`, orchestration |
| `KernelWriterAssembly.py` | Low-level GCN assembly emission for the GEMM main loop |
| `KernelWriterBetaOnly.py` | Kernel writer for beta-scaling-only kernels |
| `KernelWriterConversion.py` | Kernel writer for data-type conversion kernels |
| `KernelWriterReduction.py` | Kernel writer for reduction helper kernels |
| `KernelWriterModules.py` | Reusable assembly module helpers shared across writers |
| `KernelWriterActivationEnumHeader.py` | Generates C++ header for activation function enum |
| `KernelWriterActivationFunction.py` | Generates assembly for standalone activation functions |
| `KernelHelperNaming.py` | `KernelHelperEnum`, naming helpers for helper kernels |
| `Activation.py` | `ActivationType` enum; assembly code generation for activation functions |
| `AsmAddressCalculation.py` | Assembly address calculation helpers |
| `AsmMemoryInstruction.py` | `MemoryInstruction` abstraction for load/store instructions |
| `AsmStoreState.py` | State tracking for the assembly store path |
| `GenerateSummations.py` | Summation index loop generation |
| `ClientWriter.py` | Writes and runs the benchmark client executable |
| `ClientExecutable.py` | Locates and invokes the pre-built benchmark client binary |
| `TensileClientConfig.py` | Config data class for the benchmark client |
| `Configuration.py` | Parses and validates Tensile YAML configuration files |
| `CustomKernels.py` | Support for user-supplied custom kernel YAML files |
| `CustomYamlLoader.py` | Extended YAML loader (handles include directives, etc.) |
| `EmbeddedData.py` | Embeds binary data (code objects) into C++ source as hex arrays |
| `ParallelExecution.py` | Utilities for parallel kernel compilation with joblib |
| `TensileBenchmarkCluster.py` | Cluster-mode benchmarking coordination |
| `TensileBenchmarkClusterScripts.py` | Scripts for cluster job submission |
| `TensileBenchmarkLibraryClient.py` | Library-mode benchmark client orchestration |
| `TensileLibLogicToYaml.py` | Converts library logic binary files back to YAML |
| `TensileMergeLibrary.py` | Merges multiple solution libraries into one |
| `TensileRetuneLibrary.py` | Re-tunes an existing library with new benchmark data |
| `TensileUpdateLibrary.py` | Updates a library by adding new solutions |

---

## Workflow Entry Points

### Running a Full Auto-Tune

```
Tensile/bin/Tensile <config.yaml> <output_dir>
```

This calls `Tensile.Tensile()` → `executeStepsInConfig()`, which runs the three phases
sequentially. The `config.yaml` describes the problem type, parameter ranges, and problem sizes
to benchmark.

### Generating Kernels Without Benchmarking

Pass `buildOnly=True` to `executeStepsInConfig()`. This compiles kernels but skips the hardware
benchmarking step.

### Library Maintenance Utilities

- `TensileMergeLibrary.py` — merge libraries from different architectures or problem types.
- `TensileRetuneLibrary.py` — pick the best solutions from an existing library for a target device.
- `TensileUpdateLibrary.py` — extend an existing library with new solutions.
- `TensileLibLogicToYaml.py` — convert binary logic files to human-readable YAML for inspection.

---

## Output Directory Structure

A typical Tensile run produces:

```
<output_dir>/
  build_tmp/                  Compiled kernels and objects
  1_BenchmarkProblems/        Per-step benchmark configuration
  2_BenchmarkData/            Raw benchmark CSV results
  3_LibraryLogic/             YAML logic files (solution selection tables)
  4_LibraryClient/            Final compilable C++ library + code objects
```
