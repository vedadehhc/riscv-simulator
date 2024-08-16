# New RISC-V Test Program with .data Section
# This program interacts with preloaded data and modifies it.

.text
.globl _start

_start:
    # Load value from .data section into registers
    lw x1, 0x40(x0)  # Load data at address 0x40 into x1
    lw x2, 0x44(x0)  # Load data at address 0x44 into x2

    # Perform arithmetic operation
    add x3, x1, x2    # x3 = x1 + x2

    # Store result back in data section
    sw x3, 72(x0)  # Store result at address 0x48 (0x40 + 8)

    # Exit program
    ebreak

.data
    .data_offset = 0x40  # Specify data offset (64 in decimal)
    # Preloaded data
    .word 42          # Data value 1 at offset 0x40
    .word 58          # Data value 2 at offset 0x44
    .word 0           # Result storage at offset 0x48
