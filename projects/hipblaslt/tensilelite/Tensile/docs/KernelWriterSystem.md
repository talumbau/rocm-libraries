# Kernel Writer System

## What It Is

The Kernel Writer system is the part of Tensile responsible for generating GPU kernel source
code — primarily AMD GCN assembly — from a `Solution` parameter dictionary. Each writer class
produces the text of one or more GPU kernels that will be assembled and linked into a code object
file.

---

## Class Hierarchy

There are two parallel abstract base classes in this system:

```
KernelWriterBase  (abstract ABC, KernelWriterBase.py)
  ├── KernelWriterBetaOnly  (KernelWriterBetaOnly.py)
  ├── KernelWriterConversion  (KernelWriterConversion.py)
  ├── KernelWriterReduction  (KernelWriterReduction.py)
  ├── KernelWriterActivationEnumHeader  (KernelWriterActivationEnumHeader.py)
  └── KernelWriterActivationFunction  (KernelWriterActivationFunction.py)

KernelWriter  (abstract, metaclass=abc.ABCMeta, KernelWriter.py)
  └── KernelWriterAssembly  (KernelWriterAssembly.py)
```

`KernelWriter` and `KernelWriterBase` are **independent** abstract classes. `KernelWriterBase`
is the base for helper kernels (beta-only, conversion, reduction, activation), while
`KernelWriter` is the base for the full GEMM assembly kernel via `KernelWriterAssembly`.

---

## KernelWriterBase (`KernelWriterBase.py`)

`KernelWriterBase` is an abstract base class that defines the interface all kernel writers must
satisfy. It also holds a small set of string constants used in generated code (HIP intrinsic
names, type strings, etc.).

**Abstract methods every subclass must implement:**

| Method | Purpose |
|--------|---------|
| `getKernelName()` | Returns the unique name for this kernel |
| `getHeaderFileString()` | Returns the content of the `.h` file for the kernel |
| `getSourceFileString()` | Returns the content of the `.cpp` / `.s` source file |

**Constants defined in `KernelWriterBase.py`:**
- `KERNEL_HELPER_FILENAME_CPP = "Kernels.cpp"` — output filename for helper kernels C++ source
- `KERNEL_HELPER_FILENAME_H = "Kernels.h"` — output filename for helper kernels header

The class also implements a dict-like interface (`keys()`, `__len__`, `__iter__`, `__getitem__`,
`__setitem__`) so that a writer object can be used like a solution dictionary.

---

## KernelWriter (`KernelWriter.py`)

`KernelWriter` is the main class for the assembly GEMM kernel. It extends `KernelWriterBase`
and provides the high-level orchestration of register allocation, loop structure, address
calculation, and instruction emission.

### Key Data Classes

**`ConstValues` (frozen dataclass)**
Holds compile-time constants used during code generation:
- `initLdsValue = 0xFFFFFFFF` — value written to LDS during initialization
- `initSgprValue = 0x0` — SGPR initialization value
- `initVgprValue = 0xFFFFFFFF` — VGPR initialization value
- `maxOccupancy = 10` — maximum wave occupancy
- `ldsOOB = 0xF00000` — out-of-bounds LDS marker for debugging

**`MatrixInfo` (dataclass)**
Tracks VGPR allocation for a matrix operand (A or B):
- `numVgprValu`, `numVgprValuPack`, `startVgprValu` — VGPR counts/offsets for value storage

### Interaction with rocIsa

`KernelWriter` imports heavily from the `rocisa` package, which provides:
- IR nodes (`Module`, `TextBlock`, `KernelBody`) for building an in-memory assembly tree
- Register containers (`RegisterContainer`, `VCC`, `EXEC`, `vgpr`, `sgpr`, `accvgpr`)
- Concrete instruction classes (`MFMAInstruction`, `BufferLoadB128`, `DSStoreB64`, etc.)
- A `RegisterPool` for managing register allocation
- `LabelManager` for creating unique branch labels
- `rocIsaPass` / `rocIsaPassOption` for post-processing passes

The writer builds a tree of `Module` and instruction nodes, then serializes it to assembly text.

---

## KernelWriterAssembly (`KernelWriterAssembly.py`)

`KernelWriterAssembly` is the low-level assembly emitter that produces the actual `.s` files
for the GEMM kernel. It extends `KernelWriter` and emits instruction sequences for:

- Global memory loads (buffer/flat instructions)
- Local Data Share (LDS) loads and stores
- Matrix Fused Multiply-Add instructions (MFMA / WMMA)
- Address calculation for tiles
- Loop control flow (prefetch, pipelining, unrolling)
- Store path with optional beta scaling and activation

It imports a large number of instruction classes from `rocisa.instruction` and helper functions
from `rocisa.functions` (e.g., `vectorStaticDivide`, `scalarStaticDivideAndRemainder`,
`ArgumentLoader`).

`KernelWriterAssembly` also uses the `Component` plugin system to select hardware-specific
instruction sequences (e.g., MAC / MFMA variants) at generation time by calling
`Component.<subtype>.find(self)`.

---

## KernelWriterBetaOnly (`KernelWriterBetaOnly.py`)

Generates a helper kernel that performs only the beta-scaling step of GEMM:

```
C[i] = beta * C[i]
```

This is needed when a GEMM kernel wants to defer or skip the alpha/beta scaling in the main
kernel and apply it separately. The kernel name and its parameters are derived from the main
solution's `ProblemType`.

---

## KernelWriterConversion (`KernelWriterConversion.py`)

Generates a helper kernel that converts data from one type to another (e.g., `fp32` → `fp16`).
Used when the main kernel accumulates in a higher-precision type and the output requires a
different data type.

---

## KernelWriterReduction (`KernelWriterReduction.py`)

Generates a helper kernel for the GlobalSplitU (GSU) reduction step. When a GEMM is split
across multiple workgroups along the K dimension (`GlobalSplitU > 1`), the partial results from
each workgroup must be reduced into the final output. This kernel performs that reduction.

---

## KernelWriterModules (`KernelWriterModules.py`)

Contains reusable assembly sub-module generators shared across multiple writer classes:
- Prologue/epilogue code shared between GEMM, beta-only, and reduction kernels
- Store path helpers
- Common address computation snippets

Imported via `from .KernelWriterModules import *` into `KernelWriter.py`.

---

## KernelWriterActivationEnumHeader (`KernelWriterActivationEnumHeader.py`)

Generates a C++ header file (not a GPU kernel) that defines an `enum class` for supported
activation function types. This header is included by both the generated GEMM kernel and the
host-side library.

---

## KernelWriterActivationFunction (`KernelWriterActivationFunction.py`)

Generates a standalone GPU kernel that applies an activation function (ReLU, GELU, sigmoid,
etc.) to a buffer. This is used when activation is applied as a separate kernel pass rather than
being fused into the GEMM epilogue.

The actual activation assembly is generated by the `Activation` module (`Activation.py`).

---

## KernelHelperNaming (`KernelHelperNaming.py`)

Provides:
- `KernelHelperEnum` — an enum identifying which helper kernel types exist (beta-only,
  conversion, reduction).
- `initHelperKernelObjects()` — instantiates the appropriate set of helper writers for a given
  solution.

---

## Helper Kernel Lifecycle

1. `BenchmarkProblems.py` calls `writeSolutionsAndKernels()` (from `TensileCreateLibrary.py`),
   which iterates over solutions and calls the appropriate writer for each.
2. Each writer's `getSourceFileString()` produces the raw assembly (`.s`) or C++ (`.cpp`) text.
3. `buildAssemblyCodeObjectFiles()` in `Toolchain/Assembly.py` assembles and links the `.s`
   files into `.co` code objects.
4. `buildSourceCodeObjectFiles()` in `Toolchain/Source.py` compiles the `.cpp` files.
