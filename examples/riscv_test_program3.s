# RISC-V Test Program 3
# This program tests branch and jump instructions.

.text
.globl _start

_start:
    li x1, 0          # Initialize x1 to 0
    li x2, 10         # Initialize x2 to 10
    li x3, 1          # Initialize x3 to 1 (increment by 1)

    compare:
    beq x1, x2, end  # Branch to 'end' if x1 == x2
    jal x4, increment    # Jump and link to 'increment'

    increment:
    add x1, x1, x3   # x1 = x1 + x3
    jal x4, compare  # Jump back to 'compare'

    end:
    ebreak

.data
