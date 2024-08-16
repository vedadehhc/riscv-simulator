import sys
import re
import struct

# Define constants
DEFAULT_MEMORY_SIZE = 4 * 1024  # 4KB default memory size
NUM_REGISTERS = 32
DEFAULT_DATA_OFFSET = 1024  # Default offset for .data section

def parse_assembly(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    text_section = []
    data_section = []
    current_section = None
    data_offset = DEFAULT_DATA_OFFSET
    labels = {}
    current_address = 0

    for line in lines:
        # Remove comments and strip whitespace
        line = re.sub(r'#.*', '', line).strip()

        if line.startswith('.text'):
            current_section = text_section
            current_address = 0
        elif line.startswith('.data'):
            current_section = data_section
        elif line.startswith('.data_offset'):
            _, offset = line.split('=')
            data_offset = int(offset.strip(), 0)  # 0 base allows for hex values
        elif line.startswith('.word'):
            if current_section == data_section:
                value = int(line.split()[-1], 0)  # Parse the value, supporting hex
                data_section.append(value)
        elif line.endswith(':'):
            # This is a label
            label = line[:-1].strip()
            labels[label] = current_address
        elif line and not line.startswith('.'):
            if current_section == text_section:
                text_section.append(line)
                current_address += 4  # Each instruction is 4 bytes

    return text_section, data_section, data_offset, labels

def encode_instruction(instruction, labels, current_address):
    parts = instruction.split()
    opcode = parts[0].lower()
    print(f"Debug: Encoding instruction: {instruction}, current_address: 0x{current_address:x}")
    print(f"Debug: Labels: {labels}")

    def parse_register(reg):
        reg = reg.strip().rstrip(',')
        if not reg.startswith('x') or not reg[1:].isdigit():
            raise ValueError(f"Invalid register format: {reg}")
        return int(reg[1:])

    def parse_immediate(imm, address):
        if imm in labels:
            offset = labels[imm] - address
            print(f"Debug: Label {imm} resolved to offset 0x{offset:x}")
            return offset
        try:
            value = int(imm, 0)  # Use base 0 to handle both decimal and hex
            print(f"Debug: Immediate parsed as 0x{value:x}")
            return value
        except ValueError:
            raise ValueError(f"Invalid immediate value or undefined label: {imm}")

    if opcode == 'li':  # Load Immediate (pseudo-instruction)
        rd = parse_register(parts[1])
        imm = parse_immediate(parts[2], current_address)
        return (imm << 20) | (rd << 7) | 0x13  # ADDI instruction
    elif opcode in ['add', 'sub', 'and', 'or']:
        rd = parse_register(parts[1])
        rs1 = parse_register(parts[2])
        rs2 = parse_register(parts[3])
        funct3 = {'add': 0, 'sub': 0, 'and': 7, 'or': 6}[opcode]
        funct7 = 0x20 if opcode == 'sub' else 0
        return (funct7 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | 0x33
    elif opcode == 'sw':
        rs2 = parse_register(parts[1])
        offset, rs1 = parts[2].split('(')
        rs1 = parse_register(rs1[:-1])  # Remove ')'
        imm = parse_immediate(offset, current_address)
        return ((imm & 0xFE0) << 20) | (rs2 << 20) | (rs1 << 15) | ((imm & 0x1F) << 7) | 0x23
    elif opcode == 'lw':
        rd = parse_register(parts[1])
        offset, rs1 = parts[2].split('(')
        rs1 = parse_register(rs1[:-1])  # Remove ')'
        imm = parse_immediate(offset, current_address)
        return (imm << 20) | (rs1 << 15) | (2 << 12) | (rd << 7) | 0x03
    elif opcode == 'jal':
        if len(parts) == 3:
            rd = parse_register(parts[1])
            imm = parse_immediate(parts[2], current_address)
        else:
            rd = 1  # Default to x1 for return address
            imm = parse_immediate(parts[1], current_address)
        print(f"Debug: JAL target address: 0x{imm:x}")
        imm -= current_address  # Calculate relative offset
        print(f"Debug: JAL relative offset: 0x{imm:x}")
        # Ensure imm is sign-extended to 21 bits
        imm = ((imm + 0x100000) & 0x1FFFFF) - 0x100000
        encoded = ((imm & 0x100000) << 11) | ((imm & 0xFF000)) | \
                  ((imm & 0x800) << 9) | ((imm & 0x7FE) << 20) | (rd << 7) | 0x6F
        print(f"Debug: JAL encoded instruction: 0x{encoded:08x}")
        return encoded
    elif opcode == 'jalr':
        rd = parse_register(parts[1])
        if '(' in parts[2]:
            offset, rs1 = parts[2].split('(')
            rs1 = parse_register(rs1[:-1])  # Remove ')'
            imm = parse_immediate(offset, current_address)
        else:
            rs1 = parse_register(parts[2])
            imm = 0
        # Ensure imm is sign-extended to 12 bits
        imm = ((imm + 0x800) & 0xFFF) - 0x800
        return (imm << 20) | (rs1 << 15) | (0 << 12) | (rd << 7) | 0x67
    elif opcode in ['beq', 'bne', 'blt', 'bge', 'bltu', 'bgeu']:
        rs1 = parse_register(parts[1])
        rs2 = parse_register(parts[2])
        imm = parse_immediate(parts[3], current_address)
        print(f"Debug: Branch target address: 0x{imm:x}")
        imm -= current_address  # Calculate relative offset
        print(f"Debug: Branch relative offset: 0x{imm:x}")
        # Ensure imm is sign-extended to 13 bits
        imm = ((imm + 0x1000) & 0x1FFF) - 0x1000
        funct3 = {'beq': 0, 'bne': 1, 'blt': 4, 'bge': 5, 'bltu': 6, 'bgeu': 7}[opcode]
        encoded = ((imm & 0x1000) << 19) | ((imm & 0x7E0) << 20) | (rs2 << 20) | \
                  (rs1 << 15) | (funct3 << 12) | ((imm & 0x1E) << 7) | ((imm & 0x800) >> 4) | 0x63
        print(f"Debug: Branch encoded instruction: 0x{encoded:08x}")
        return encoded
    elif opcode == 'ebreak':
        return 0x00100073
    else:
        raise ValueError(f"Unsupported instruction: {instruction}")

class RISCVSimulator:
    def __init__(self, memory_size=DEFAULT_MEMORY_SIZE, data_offset=DEFAULT_DATA_OFFSET):
        self.memory_size = memory_size
        self.memory = bytearray(memory_size)
        self.registers = [0] * NUM_REGISTERS
        self.pc = 0  # Program Counter
        self.data_offset = data_offset
        self.text_size = 0  # Will be set when loading the program

    def load_program(self, filename):
        text_instructions, data_values, file_data_offset, labels = parse_assembly(filename)

        # Update data_offset if specified in the file
        if file_data_offset != DEFAULT_DATA_OFFSET:
            self.data_offset = file_data_offset
        print(f"Data offset set to: 0x{self.data_offset:x}")

        # Load .text section
        for i, instruction in enumerate(text_instructions):
            encoded = encode_instruction(instruction, labels, i*4)
            self.memory[i*4:i*4+4] = encoded.to_bytes(4, byteorder='little')
            print(f"Loaded instruction at 0x{i*4:x}: {encoded:08x}")

        self.text_size = len(text_instructions) * 4
        print(f"Text section size: {self.text_size} bytes")

        # Load .data section
        print("Loading .data section:")
        for i, value in enumerate(data_values):
            data_addr = self.data_offset + i*4
            if data_addr + 4 > self.memory_size:
                raise ValueError("Data section exceeds memory size")
            value_int = value  # Use the value directly, as it's already an integer
            self.memory[data_addr:data_addr + 4] = value_int.to_bytes(4, byteorder='little', signed=True)
            print(f"  Loaded 0x{value_int:08x} at address 0x{data_addr:x}")
        print(f"Data section loaded at offset 0x{self.data_offset:x}")

        # Store labels for later use
        self.labels = {}
        for label, address in labels.items():
            self.labels[label] = address
        print("Labels:")
        for label, address in self.labels.items():
            print(f"  {label}: 0x{address:x}")

        # Ensure data is correctly loaded by printing memory contents
        print("Memory contents after loading:")
        for i in range(self.data_offset, self.data_offset + len(data_values) * 4, 4):
            value = int.from_bytes(self.memory[i:i+4], byteorder='little', signed=True)
            print(f"  Address 0x{i:x}: 0x{value:08x}")

    def execute_instruction(self):
        instruction = int.from_bytes(self.memory[self.pc:self.pc+4], byteorder='little')
        opcode = instruction & 0x7F
        rd = (instruction >> 7) & 0x1F
        funct3 = (instruction >> 12) & 0x7
        rs1 = (instruction >> 15) & 0x1F
        rs2 = (instruction >> 20) & 0x1F
        imm = instruction >> 20

        print(f"Executing instruction at PC 0x{self.pc:08x}: {instruction:032b}")
        print(f"Opcode: {opcode:07b}, rd: {rd}, rs1: {rs1}, rs2: {rs2}, funct3: {funct3:03b}, imm: {imm}")

        if opcode == 0x13:  # ADDI
            imm = (imm - 2**12) if imm & 0x800 else imm  # Sign extend
            self.registers[rd] = (self.registers[rs1] + imm) & 0xFFFFFFFF
            print(f"ADDI: x{rd} = x{rs1} ({self.registers[rs1]}) + {imm} = {self.registers[rd]}")
        elif opcode == 0x33:  # ADD, SUB, AND, OR
            if funct3 == 0x0:
                if (instruction >> 25) == 0x20:  # SUB
                    self.registers[rd] = (self.registers[rs1] - self.registers[rs2]) & 0xFFFFFFFF
                    print(f"SUB: x{rd} = x{rs1} ({self.registers[rs1]}) - x{rs2} ({self.registers[rs2]}) = {self.registers[rd]}")
                else:  # ADD
                    print(f"DEBUG: Before ADD - x{rs1}: {self.registers[rs1]}, x{rs2}: {self.registers[rs2]}")
                    self.registers[rd] = (self.registers[rs1] + self.registers[rs2]) & 0xFFFFFFFF
                    print(f"ADD: x{rd} = x{rs1} ({self.registers[rs1]}) + x{rs2} ({self.registers[rs2]}) = {self.registers[rd]}")
                    print(f"DEBUG: After ADD - x{rd}: {self.registers[rd]}")
            elif funct3 == 0x7:  # AND
                self.registers[rd] = self.registers[rs1] & self.registers[rs2]
                print(f"AND: x{rd} = x{rs1} ({self.registers[rs1]}) & x{rs2} ({self.registers[rs2]}) = {self.registers[rd]}")
            elif funct3 == 0x6:  # OR
                self.registers[rd] = self.registers[rs1] | self.registers[rs2]
                print(f"OR: x{rd} = x{rs1} ({self.registers[rs1]}) | x{rs2} ({self.registers[rs2]}) = {self.registers[rd]}")
        elif opcode == 0x23:  # SW
            imm = ((instruction >> 25) << 5) | ((instruction >> 7) & 0x1F)
            imm = (imm - 2**12) if imm & 0x800 else imm  # Sign extend
            addr = (self.registers[rs1] + imm) & 0xFFFFFFFF
            value = self.registers[rs2]
            print(f"SW: Storing value {value} (0x{value:08x}) to address {addr} (0x{addr:08x})")
            print(f"SW: Base address: {self.registers[rs1]}, Immediate: {imm}")
            self.memory[addr:addr+4] = value.to_bytes(4, byteorder='little')
            print(f"SW: Memory at {addr} (0x{addr:08x}) is now: {self.memory[addr:addr+4].hex()}")
        elif opcode == 0x03:  # LW
            imm = instruction >> 20
            imm = (imm - 2**12) if imm & 0x800 else imm  # Sign extend
            addr = (self.registers[rs1] + imm) & 0xFFFFFFFF
            print(f"LW: Loading from address {addr} (0x{addr:08x})")
            print(f"LW: Base address: {self.registers[rs1]}, Immediate: {imm}")
            value_bytes = self.memory[addr:addr+4]
            self.registers[rd] = int.from_bytes(value_bytes, byteorder='little')
            print(f"LW: Loaded bytes: {value_bytes.hex()}, Interpreted as: {self.registers[rd]}")
        elif opcode == 0x6F:  # JAL
            imm = ((instruction & 0x80000000) >> 11) | (instruction & 0xFF000) | \
                  ((instruction >> 9) & 0x800) | ((instruction >> 20) & 0x7FE)
            imm = (imm - 2**21) if imm & 0x100000 else imm  # Sign extend
            self.registers[rd] = self.pc + 4
            old_pc = self.pc
            self.pc = (self.pc + imm) & 0xFFFFFFFF
            print(f"DEBUG: JAL - Old PC: 0x{old_pc:08x}, New PC: 0x{self.pc:08x}")
            print(f"JAL: Jump from 0x{old_pc:08x} to 0x{self.pc:08x}, store return address {self.registers[rd]:08x} in x{rd}")
            print(f"JAL: Immediate value: {imm}, New PC: 0x{self.pc:08x}")
            return False  # Don't increment PC after this instruction
        elif opcode == 0x67:  # JALR
            imm = instruction >> 20
            imm = (imm - 2**12) if imm & 0x800 else imm  # Sign extend
            target = (self.registers[rs1] + imm) & 0xFFFFFFFE  # Clear least significant bit
            self.registers[rd] = self.pc + 4
            old_pc = self.pc
            self.pc = target
            print(f"JALR: Jump from 0x{old_pc:08x} to (x{rs1} ({self.registers[rs1]:08x}) + {imm}) & ~1 = 0x{self.pc:08x}, store return address {self.registers[rd]:08x} in x{rd}")
            return False  # Don't increment PC after this instruction
        elif opcode == 0x63:  # Branch instructions
            imm = ((instruction & 0x80000000) >> 19) | ((instruction & 0x80) << 4) | \
                  ((instruction >> 20) & 0x7E0) | ((instruction >> 7) & 0x1E)
            imm = (imm - 2**13) if imm & 0x1000 else imm  # Sign extend
            cond = False
            if funct3 == 0x0:  # BEQ
                cond = self.registers[rs1] == self.registers[rs2]
                print(f"BEQ: Branch if x{rs1} ({self.registers[rs1]:08x}) == x{rs2} ({self.registers[rs2]:08x})")
            elif funct3 == 0x1:  # BNE
                cond = self.registers[rs1] != self.registers[rs2]
                print(f"BNE: Branch if x{rs1} ({self.registers[rs1]:08x}) != x{rs2} ({self.registers[rs2]:08x})")
            elif funct3 == 0x4:  # BLT
                cond = ((self.registers[rs1] ^ 0x80000000) - (self.registers[rs2] ^ 0x80000000)) < 0
                print(f"BLT: Branch if x{rs1} ({self.registers[rs1]:08x}) < x{rs2} ({self.registers[rs2]:08x}) (signed)")
            elif funct3 == 0x5:  # BGE
                cond = ((self.registers[rs1] ^ 0x80000000) - (self.registers[rs2] ^ 0x80000000)) >= 0
                print(f"BGE: Branch if x{rs1} ({self.registers[rs1]:08x}) >= x{rs2} ({self.registers[rs2]:08x}) (signed)")
            elif funct3 == 0x6:  # BLTU
                cond = (self.registers[rs1] & 0xFFFFFFFF) < (self.registers[rs2] & 0xFFFFFFFF)
                print(f"BLTU: Branch if x{rs1} ({self.registers[rs1]:08x}) < x{rs2} ({self.registers[rs2]:08x}) (unsigned)")
            elif funct3 == 0x7:  # BGEU
                cond = (self.registers[rs1] & 0xFFFFFFFF) >= (self.registers[rs2] & 0xFFFFFFFF)
                print(f"BGEU: Branch if x{rs1} ({self.registers[rs1]:08x}) >= x{rs2} ({self.registers[rs2]:08x}) (unsigned)")
            old_pc = self.pc
            if cond:
                self.pc = (self.pc + imm) & 0xFFFFFFFF
                print(f"Branch taken, jumping from 0x{old_pc:08x} to 0x{self.pc:08x}")
                print(f"Branch: Immediate value: {imm}, New PC: 0x{self.pc:08x}")
                return False  # Don't increment PC after this instruction
            else:
                print(f"Branch not taken, PC remains at 0x{self.pc:08x}")
        elif opcode == 0x73 and funct3 == 0x0:  # ecall, ebreak
            if (instruction >> 20) & 0xFFF == 0x001:  # ebreak
                print("EBREAK instruction encountered. Terminating execution.")
                return True  # Indicating termination signal

        self.pc += 4
        print(f"Updated PC: 0x{self.pc:08x}")
        return False  # Continue execution

    def run(self):
        instruction_count = 0
        cycle_count = 0
        max_cycles = 100000  # Safeguard against infinite loops
        print("Starting simulation...")
        try:
            while self.pc < len(self.memory) and cycle_count < max_cycles:
                print(f"\nCycle {cycle_count}:")
                print(f"PC: 0x{self.pc:08x}")
                print("Register state:")
                for i in range(0, 32, 4):
                    print(f"x{i:2d}-x{i+3:2d}: " + " ".join(f"{self.registers[j]:08x}" for j in range(i, i+4)))

                # Ensure x0 is always zero
                self.registers[0] = 0

                instruction = int.from_bytes(self.memory[self.pc:self.pc+4], byteorder='little')
                print(f"Instruction: 0x{instruction:08x}")

                old_pc = self.pc
                try:
                    if self.execute_instruction():
                        print(f"EBREAK encountered. Terminating after {instruction_count + 1} instructions and {cycle_count + 1} cycles.")
                        break  # Exit if ebreak is encountered
                except ValueError as e:
                    print(f"Illegal instruction encountered: {str(e)}")
                    print(f"Terminating after {instruction_count} instructions and {cycle_count} cycles.")
                    break

                instruction_count += 1
                cycle_count += 1

                if self.pc == old_pc:
                    print(f"Warning: PC did not change after instruction execution. This may indicate an infinite loop.")
                    user_input = input("Press Enter to continue, 'q' to quit, or 's' to skip to next instruction: ")
                    if user_input.lower() == 'q':
                        print("Simulation aborted by user.")
                        break
                    elif user_input.lower() == 's':
                        self.pc += 4
                        print(f"Skipped to next instruction. New PC: 0x{self.pc:08x}")

                if self.pc >= len(self.memory):
                    print(f"End of memory reached. Terminating after {instruction_count} instructions and {cycle_count} cycles.")
                    break

                if cycle_count % 1000 == 0:
                    user_input = input(f"Executed {cycle_count} cycles. Press Enter to continue or 'q' to quit: ")
                    if user_input.lower() == 'q':
                        print("Simulation aborted by user.")
                        break

            if cycle_count >= max_cycles:
                print(f"Maximum cycle limit ({max_cycles}) reached. Possible infinite loop.")

            print(f"\nSimulation statistics:")
            print(f"  Instructions executed: {instruction_count}")
            print(f"  Cycles executed: {cycle_count}")
            if instruction_count > 0:
                print(f"  Average CPI (Cycles Per Instruction): {cycle_count / instruction_count:.2f}")
            else:
                print("  Average CPI: N/A (No instructions executed)")

        except KeyboardInterrupt:
            print("\nSimulation aborted by user.")
        except Exception as e:
            print(f"Error during execution: {str(e)}")
        finally:
            print("Simulation completed.")
            self.dump_state()  # Always dump the final state

    def debug_output(self):
        print(f"PC: {self.pc}")
        for i, reg in enumerate(self.registers):
            print(f"x{i}: {reg}")
        print("Memory:")
        for i in range(0, len(self.memory), 16):
            print(f"{i:04x}: {' '.join(f'{b:02x}' for b in self.memory[i:i+16])}")

    def dump_state(self):
        with open("state_dump.txt", "w") as f:
            f.write("Register State:\n")
            for i, reg in enumerate(self.registers):
                f.write(f"x{i}: {reg}\n")

            f.write("\nMemory State:\n")
            f.write(".text section:\n")
            for i in range(0, self.data_offset, 16):
                f.write(f"{i:04x}: {' '.join(f'{b:02x}' for b in self.memory[i:i+16])}\n")

            f.write("\n.data section:\n")
            for i in range(self.data_offset, len(self.memory), 16):
                f.write(f"{i:04x}: {' '.join(f'{b:02x}' for b in self.memory[i:i+16])}\n")

def main():
    import argparse

    parser = argparse.ArgumentParser(description="RISC-V Simulator")
    parser.add_argument("program_file", help="Path to the RISC-V assembly program file")
    parser.add_argument("--memory-size", type=int, default=DEFAULT_MEMORY_SIZE,
                        help="Size of memory in bytes (default: %(default)s)")
    parser.add_argument("--data-offset", type=int, default=DEFAULT_DATA_OFFSET,
                        help="Offset for .data section in bytes (default: %(default)s)")

    args = parser.parse_args()

    simulator = RISCVSimulator(args.memory_size, args.data_offset)
    simulator.load_program(args.program_file)
    simulator.run()
    simulator.dump_state()

if __name__ == "__main__":
    main()
