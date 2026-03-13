# Tensile Toolchain Overview

## What This Package Does

The `Tensile.Toolchain` subpackage wraps the ROCm command-line build tools — assembler, HIP
compiler, offload bundler, and linker — in typed Python objects. Tensile drives these tools to
compile generated assembly (`.s`) and HIP C++ (`.cpp`) source files into AMD GPU code object
files (`.co` / `.hsaco`) that can be loaded and executed at runtime.

The package deliberately keeps each tool as a thin, callable wrapper so that the rest of Tensile
can treat compilation as a function call without knowing the exact command-line interface of each
tool.

---

## Package Structure

| File | Role |
|------|------|
| `Component.py` | Defines `Assembler`, `Compiler`, `Linker`, `Bundler` tool wrapper classes and the shared `Component` base |
| `Assembly.py` | `AssemblyToolchain` named tuple + `buildAssemblyCodeObjectFiles()` pipeline function |
| `Source.py` | `SourceToolchain` named tuple + `buildSourceCodeObjectFiles()` pipeline function |
| `Validators.py` | `ToolchainDefaults` constants + `validateToolchain()` path resolution function |
| `__init__.py` | Empty — package marker |

---

## Tool Wrapper Classes (`Component.py`)

All tool wrappers inherit from the shared `Component` base class.

### `Component` (base class)

Represents any ROCm toolchain executable. On construction it:

1. Calls `get_rocm_version()` (once, via a class-level attribute) using `hipconfig --version`.
2. Calls `_getVersion()` on the specific executable to extract its version via a regex.

Provides three properties: `path`, `version`, `rocm_version`.

```python
class Component:
    _rocm_version = get_rocm_version()

    def __init__(self, component_path: Path,
                 version_flag: str="--version",
                 version_regex: str=r"version\s+([\d.]+)"):
        ...
```

### `Assembler`

Wraps `amdclang++` (or `clang++.exe` on Windows) for assembling `.s` → `.o`.

```python
class Assembler(Component):
    def __init__(self, component_path: Path, co_version: str, debug: bool=False): ...
    def __call__(self, targetGfx: str, wavefrontSize: int, srcPath: str, destPath: str): ...
```

`__call__` builds and executes a command of the form:

```
amdclang++ -x assembler --target=amdgcn-amd-amdhsa
           -mcode-object-version=<co_version>
           -mcpu=<targetGfx> -mwavefrontsize64|-mno-wavefrontsize64
           -c <srcPath> -o <destPath>
```

The `Tensile_ASM_COMPILER_LAUNCHER` environment variable, if set, is prepended to the command
(useful for tools like `ccache` or `distcc`).

### `Compiler`

Wraps `amdclang++` for compiling HIP C++ source `.cpp` → `.o` (device-only).

```python
class Compiler(Component):
    def __init__(self, compiler_path: Path, build_id_kind: str,
                 asan_build: bool=False, save_temps: bool=False): ...
    def __call__(self, include_path: str, target_list: List[str],
                 srcPath: str, destPath: str): ...
```

`__call__` invokes:

```
amdclang++ -D__HIP_HCC_COMPAT_MODE__=1 --offload-device-only -x hip -O3
           -Xoffload-linker --build-id=<build_id_kind> -std=c++17
           -I <include_path>
           --offload-arch=<gfx1> --offload-arch=<gfx2> ...
           -c <srcPath> -o <destPath>
```

The `Tensile_CXX_COMPILER_LAUNCHER` environment variable is respected similarly.

### `Linker`

Wraps `amdclang++` for linking multiple `.o` → a single code object `.co`.

```python
class Linker(Component):
    def __init__(self, linker_path: Path, build_id_kind: str): ...
    def __call__(self, srcPaths: List[str], destPath: str): ...
```

When the total argument length would exceed the OS limit, the linker uses a **response file**
(`clang_args.txt` written in the current directory) instead of passing source paths directly on
the command line. This is always the case on Windows (8191-char limit) and conditionally on
POSIX (checked against `sysconf("SC_ARG_MAX")`).

### `Bundler`

Wraps `clang-offload-bundler` for bundling/unbundling and compressing code objects.

```python
class Bundler(Component):
    def __init__(self, bundler_path: Path): ...
    def targets(self, objFile: str) -> List[str]: ...        # list embedded target triples
    def compress(self, srcPath: str, destPath: str, target: str): ...  # .co.raw -> .co
    def __call__(self, target: str, srcPath: str, destPath: str): ...  # unbundle
```

`compress()` calls `clang-offload-bundler --compress`, embedding both a host placeholder and
the GPU code object into a single bundled `.co` file with 4096-byte alignment.

---

## Assembly Pipeline (`Assembly.py`)

### `AssemblyToolchain` (NamedTuple)

```python
class AssemblyToolchain(NamedTuple):
    assembler: Assembler
    linker: Linker
    bundler: Bundler
```

Created via:

```python
def makeAssemblyToolchain(assembler_path, bundler_path, co_version,
                          build_id_kind="sha1", debug=False) -> AssemblyToolchain:
```

### `buildAssemblyCodeObjectFiles(linker, bundler, kernels, destDir, asmDir, compress=True)`

The main pipeline function for assembly kernels. Given a list of `Solution` objects and a
directory of pre-assembled `.s` files, it:

1. Groups kernels by ISA architecture (`k['ISA']`).
2. For each architecture, collects the corresponding `.o` object files.
3. Calls `linker(objectFiles, coFileRaw)` to link them into a raw `.co.raw` code object.
4. If `compress=True`, calls `bundler.compress(coFileRaw, coFile, gfx)` to produce the final
   compressed `.co`.
5. Returns a list of final `.co` file paths.

Kernels with a `codeObjectFile` key are linked into their own named code object rather than the
default `TensileLibrary_<gfx>.co`.

---

## Source Pipeline (`Source.py`)

### `SourceToolchain` (NamedTuple)

```python
class SourceToolchain(NamedTuple):
    compiler: Compiler
    bundler: Bundler
```

Created via:

```python
def makeSourceToolchain(compiler_path, bundler_path, asan_build=False,
                        build_id_kind="sha1", save_temps=False) -> SourceToolchain:
```

### `buildSourceCodeObjectFiles(compiler, bundler, destDir, tmpObjDir, includeDir, kernelPath, cmdlineArchs)`

Compiles a single HIP C++ kernel file into per-architecture `.hsaco` code objects:

1. Calls `compiler(include_path, target_list, srcPath, destPath)` to build a fat object `.o`
   containing device code for all requested architectures.
2. Uses `bundler.targets(objFile)` to discover the embedded target triples.
3. For each target triple, calls `bundler(target, srcPath, destPath)` to unbundle the
   per-architecture `.hsaco.raw`.
4. Returns the list of created `.hsaco` paths.

`_computeSourceCodeObjectFilename()` maps a (target, base, buildPath, arch) tuple to the
correct output path, handling special naming for fallback and xnack variants.

---

## Toolchain Discovery and Validation (`Validators.py`)

### `ToolchainDefaults` (NamedTuple)

Holds the default executable names for each tool, selected by OS:

| Constant | Linux | Windows |
|----------|-------|---------|
| `CXX_COMPILER` | `amdclang++` | `clang++.exe` |
| `C_COMPILER` | `amdclang` | `clang.exe` |
| `OFFLOAD_BUNDLER` | `clang-offload-bundler` | `clang-offload-bundler.exe` |
| `ASSEMBLER` | `amdclang++` | `clang++.exe` |
| `HIP_CONFIG` | `hipconfig` | `hipconfig.exe` |
| `DEVICE_ENUMERATOR` | `amdgpu-arch` (or `rocm_agent_enumerator` on RHEL8) | `hipinfo` |

### `validateToolchain(component) -> str`

Resolves a toolchain component name to an absolute path by searching through a prioritized list
of directories:

1. `$ROCM_PATH/bin` and `$ROCM_PATH/lib/llvm/bin` (POSIX) or `$HIP_PATH/bin` (Windows)
2. `/opt/rocm/bin` and `/opt/rocm/lib/llvm/bin` (POSIX defaults)
3. `$PATH` entries

Raises `RuntimeError` if the executable cannot be found.

### Validation Helpers

```python
def supportedCCompiler(compiler: str) -> bool      # accepts amdclang or clang
def supportedCxxCompiler(compiler: str) -> bool     # accepts amdclang++ or clang++
def supportedOffloadBundler(bundler: str) -> bool   # accepts clang-offload-bundler
```

These are used at startup to ensure the user-supplied (or default) tool names are ones Tensile
knows how to work with.
