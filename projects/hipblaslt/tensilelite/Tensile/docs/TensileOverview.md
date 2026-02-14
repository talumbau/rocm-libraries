# Tensile Overview

## Purpose

Tensile is a code generation and benchmarking tool for GEMM (General Matrix Multiply) kernels on AMD GPUs. It generates optimized assembly and source kernels for various problem configurations and creates library logic for selecting the best kernel at runtime.

## Key Components

### Core Workflow Modules

- **Tensile.py** - Entry point that orchestrates the three-phase workflow:
  1. BenchmarkProblems: Runs benchmarking steps and generates performance data
  2. LibraryLogic: Analyzes benchmark data and creates logic files for kernel selection
  3. LibraryClient: Generates client callable libraries

### Configuration and Problem Definition

- **Configuration.py** - Provides `ReadWriteTransformDict` for flexible configuration management with customizable read/write transforms
- **Contractions.py** - Defines contraction operations and problem types
- **BenchmarkStructs.py** - Data structures for benchmark configurations
- **BenchmarkProblems.py** - Manages benchmark problem execution

### Kernel Generation

- **KernelWriter.py** - Base interface for kernel code generation, supporting both assembly and source kernels
- **KernelWriterAssembly.py** - Generates optimized AMD GPU assembly code using the rocisa library
- **KernelWriterConversion.py** - Handles data type conversion kernels
- **KernelWriterBetaOnly.py** - Generates specialized beta-only kernels

### Solution Selection

- **SolutionLibrary.py** - Provides library abstractions for solution selection:
  - `SingleSolutionLibrary` - Wraps a single solution
  - `MatchingLibrary` - Selects solutions based on problem properties
  - `PlaceholderLibrary` - Named placeholder for deferred loading

- **LibraryLogic.py** - Analyzes benchmark results and generates selection logic
- **SolutionSelectionLibrary.py** - Runtime kernel selection based on problem characteristics

### Hardware and ISA Support

- **Hardware.py** - Hardware capability detection and management
- **Common/Architectures.py** - ISA version handling and GPU architecture support
- **Common/Capabilities.py** - Hardware capability queries

### Activation and Memory Management

- **Activation.py** - Manages activation functions in kernels
- **AsmAddressCalculation.py** - Assembly-level address calculation for memory operations
- **AsmMemoryInstruction.py** - Memory instruction abstractions
- **AsmStoreState.py** - State management for assembly store operations

### Custom Kernels

- **CustomKernels.py** - Integration point for user-provided custom kernels
- **CustomKernels/** directory - Contains custom kernel implementations (see CustomKernels/README.md)

### Testing and Benchmarking

- **ClientWriter.py** - Generates client code for testing generated kernels
- **ClientExecutable.py** - Manages client executable configuration
- **BenchmarkSplitter.py** - Splits benchmark workloads for parallel execution

## Directory Structure

- **Common/** - Shared utilities, data types, and parameters
- **Components/** - Modular kernel components (see Components/README.md)
- **SolutionStructs/** - Solution data structures and validators
- **Tests/** - Unit and integration tests
- **Utilities/** - Helper tools and decorators
- **bin/** - Executable entry points

## Typical Workflow

1. Define problem configurations and kernel parameters
2. Run benchmarks to collect performance data
3. Analyze results to generate optimal selection logic
4. Generate library with runtime kernel selection
5. Client code calls library for efficient GEMM execution
