# RISC-V Test Program 1
# This program tests basic arithmetic, logical, and load/store operations

.text
.globl _start

_start:
    # Initialize registers
    li x1, 10      # Load immediate value 10 into x1
    li x2, 5       # Load immediate value 5 into x2

    # Arithmetic operations
    add x3, x1, x2 # x3 = x1 + x2 (should be 15)
    sub x4, x1, x2 # x4 = x1 - x2 (should be 5)

    # Logical operations
    and x5, x1, x2 # x5 = x1 & x2 (should be 0)
    or x6, x1, x2  # x6 = x1 | x2 (should be 15)

    # Store and load operations
    sw x3, 0(x0)   # Store x3 (15) at memory address 0
    lw x7, 0(x0)   # Load value from memory address 0 into x7

    # Exit program (this is a placeholder, actual exit depends on the environment)
    ebreak

.data
    # No data section needed for this simple test
