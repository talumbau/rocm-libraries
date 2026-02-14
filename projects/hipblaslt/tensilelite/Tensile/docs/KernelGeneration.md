# Kernel Generation

## Overview

Tensile generates optimized GPU kernels for GEMM operations through a flexible architecture that supports both assembly and source code generation. The kernel generation system uses a component-based approach for modularity and extensibility.

## Kernel Writer Architecture

### Base Classes

**KernelWriterBase** (KernelWriterBase.py:32-109) provides the abstract interface for all kernel writers:
- Defines platform-specific strings for GPU intrinsics
- Implements dictionary-like interface for kernel state management
- Requires subclasses to implement `getKernelName()`, `getHeaderFileString()`, and `getSourceFileString()`

**KernelWriter** (KernelWriter.py) extends the base with concrete implementations for generating kernels.

### Specialized Writers

- **KernelWriterAssembly** - Generates AMD GPU assembly using the rocisa library for maximum performance
- **KernelWriterConversion** - Creates data type conversion kernels
- **KernelWriterBetaOnly** - Generates specialized kernels for beta-only operations
- **KernelWriterReduction** - Handles reduction operations
- **KernelWriterActivationFunction** - Manages activation function integration

## Component System

### Design Philosophy

Components are modular pieces of code selected based on hardware capabilities and kernel configuration (Component.py:26-67). The class hierarchy automatically categorizes components by type (e.g., MAC components inherit from the MAC base class).

### Component Selection

Components are discovered via `Component.<subtype>.find(writer)` which searches for matching implementations. The selection process uses pattern matching against:
- **asmCaps** - Assembly instruction capabilities (e.g., `v_fma_f16`, `v_pk_fma_f16`)
- **archCaps** - Architecture-specific capabilities
- **kernel** - Kernel configuration requirements

### Matching Logic

The `PartialMatch()` function (Component.py:80-100) compares component requirements against the current context:
- Supports callable patterns for complex logic
- Recursively matches nested dictionary structures
- Enables flexible component qualification

## Assembly Generation

### Register Management

**RegisterPool** manages VGPR and SGPR allocation for assembly kernels. The system tracks:
- Available registers
- Register lifetimes
- Allocation conflicts

### Memory Operations

Assembly memory instructions are abstracted through specialized classes:
- **AsmMemoryInstruction** - Base memory instruction abstraction
- **AsmAddressCalculation** - Computes memory addresses for load/store operations
- **AsmStoreState** - Manages state for store instruction sequences

### Matrix Information Structures

**MatrixInfo** dataclasses (KernelWriter.py:76-100) track register allocation for matrix operations:
- VGPR counts for values and addresses
- SGPR counts for strides
- Specialized **ABMatrixInfo** for input matrices with additional local/global memory tracking

## Activation Functions

The **ActivationModule** (Activation.py) integrates activation functions into kernels:
- Supports various activation types (ReLU, GELU, etc.)
- Handles activation parameter passing
- Generates inline activation code

## Kernel Naming

**KernelHelperNaming** provides consistent kernel naming conventions, enabling:
- Deterministic kernel identification
- Efficient kernel lookup
- Human-readable kernel names for debugging

## File Organization

Generated kernels produce two files:
- **Kernels.h** - Header declarations
- **Kernels.cpp** - Implementation code
