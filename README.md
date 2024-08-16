# RISC-V Simulator

This project is a Python-based simulator for a subset of the RISC-V instruction set architecture (ISA). It interprets and executes RISC-V assembly code, supporting functionalities like arithmetic, logical, load/store operations, and interaction with preloaded data sections in simulated memory.

## Features
- Handles a subset of RISC-V instructions, including arithmetic, logical, branch, and memory operations.
- Supports `.text` and `.data` sections with configurable memory offset.
- Writes execution results with all register values and memory states to a file for inspection.
- Designed to facilitate learning and testing simple RISC-V programs.

## Usage
Run the simulator with a RISC-V assembly file path as input, optionally specifying memory size and data offset. For example:

```bash
python3 riscv_simulator.py riscv_test_program2.s --memory-size=128 --data-offset=64
```

## Files
- `riscv_simulator.py`: Simulator source code.
- `riscv_test_program1.s`: Initial assembly test program.
- `riscv_test_program2.s`: Test program interacting with `.data` section data.
