# Component System

## What It Is

The Component system is a plugin architecture that lets Tensile select hardware-specific
assembly code fragments at kernel generation time. Instead of a single monolithic kernel writer
full of `if/elif` chains for every GPU variant, Tensile organizes hardware-specific
implementations as **Component subclasses**. At generation time the writer asks the component
registry to find the best matching component for the current hardware and kernel configuration.

The system is defined in `Component.py`.

---

## Core Concept

A **Component** is a class that:

1. Belongs to a **category** (e.g., `MAC`, `LraTileAssignment`) by inheriting from the
   appropriate category class.
2. Declares **requirements** via class-level dictionaries:
   - `asmCaps` — required assembly capability flags (keys from the ISA capability map).
   - `archCaps` — required architecture capability flags.
   - `kernel` — required kernel parameter values (can be nested, matching solution fields like
     `ProblemType.DataType`).
3. Implements `__call__(self, writer, ...)` to generate the assembly module for that component.

Values in the requirement dictionaries can be plain literals or **lambdas** for more complex
matching logic.

---

## Key Functions

### `PartialMatch(pattern, obj, debug=False, level=0)`

Recursively checks whether `pattern` matches `obj`. A pattern is a nested dict (or scalar)
where each key must exist in `obj` with a matching value. Lambda values in the pattern are
called with the corresponding value from `obj` — they return `True` if the requirement is met.

This is the underlying matching engine used by `Component.find()`.

### `Component.find(writer)` (class method on the category base class)

Iterates over all registered subclasses of that category, calls `matches(writer)` on each, and
returns the first matching component instance. Returns `None` if no component matches, which
allows the caller to fall back to existing code.

```python
# Example usage inside a KernelWriter method:
component = Component.MAC.find(self)
if component:
    return component(self, m, innerUnroll)
# No component found — fall back to legacy code path
```

### `Component.matches(writer)` (instance method)

Checks `asmCaps`, `archCaps`, and `kernel` requirements against the writer's current hardware
and kernel settings. Returns `True` if all requirements are satisfied.

---

## LraTileProperties

A small frozen dataclass (fields currently empty in base form) that carries tile assignment
properties for the LRA (Local Result Accumulation) component type.

---

## Adding a New Component

1. Decide which category it belongs to (e.g., `Component.MAC`).
2. Create a class inheriting from that category with the required `asmCaps` / `archCaps` /
   `kernel` class attributes.
3. Override `__call__` to return the assembly module.
4. Register the file in the `__all__` list in `Tensile/Components/__init__.py`.

```python
class FMA_NonPacked(MAC):
    asmCaps = {"v_fma_f16": True, "v_pk_fma_f16": False}
    kernel = {"ProblemType": {"DataType": DataType(DataTypeEnum.Half),
                              "HighPrecisionAccumulate": False}}

    def __call__(self, writer, m, innerUnroll):
        # Return an assembly Module
        ...
```

---

## Component Categories

Component subclasses live in the `Tensile/Components/` subdirectory (one level below this
directory). Each category is defined as a nested class within `Component`:

- **`Component.MAC`** — Matrix multiply-accumulate instruction selection (MFMA, WMMA, FMA).
- **`Component.LraTileAssignment`** — Local result accumulation tile assignment.
- Additional categories added by the `Components/` files.

The categories themselves are registered when Python imports `Tensile/Components/__init__.py`.

---

## Interaction with KernelWriter

`KernelWriter` and `KernelWriterAssembly` call `Component.<Category>.find(self)` at several
points during kernel generation:

- When selecting the matrix multiply instruction for the inner loop.
- When computing tile assignment for the LRA stage.
- Potentially for other specialized code paths (load scheduling, etc.).

The `writer` object passed to `find()` provides the hardware capability maps and the current
solution's parameter dictionary, giving the component matching logic access to everything it
needs to determine compatibility.

---

## Source Files

| File | Role |
|------|------|
| `Component.py` | Base class, `PartialMatch`, `Component.find()`, `LraTileProperties` |
| `Components/` | Directory of concrete component implementations (separate subdirectory) |
