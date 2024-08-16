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

    for line in lines:
        # Remove comments and strip whitespace
        line = re.sub(r'#.*', '', line).strip()

        if line.startswith('.text'):
            current_section = text_section
        elif line.startswith('.data'):
            current_section = data_section
        elif line.startswith('.data_offset'):
            _, offset = line.split('=')
            data_offset = int(offset.strip(), 0)  # 0 base allows for hex values
        elif line.startswith('.word'):
            if current_section == data_section:
                value = int(line.split()[-1], 0)  # Parse the value, supporting hex
                data_section.append(value)
        elif line and not line.startswith('.') and not line.endswith(':'):
            if current_section == text_section:
                text_section.append(line)

    return text_section, data_section, data_offset

def encode_instruction(instruction):
    parts = instruction.split()
    opcode = parts[0].lower()

    def parse_register(reg):
        return int(reg.strip().rstrip(',')[1:])  # Remove 'x', strip whitespace and comma

    def parse_immediate(imm):
        return int(imm, 0)  # Use base 0 to handle both decimal and hex

    if opcode == 'li':  # Load Immediate (pseudo-instruction)
        rd = parse_register(parts[1])
        imm = parse_immediate(parts[2])
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
        imm = parse_immediate(offset)
        return ((imm & 0xFE0) << 20) | (rs2 << 20) | (rs1 << 15) | ((imm & 0x1F) << 7) | 0x23
    elif opcode == 'lw':
        rd = parse_register(parts[1])
        offset, rs1 = parts[2].split('(')
        rs1 = parse_register(rs1[:-1])  # Remove ')'
        imm = parse_immediate(offset)
        return (imm << 20) | (rs1 << 15) | (2 << 12) | (rd << 7) | 0x03
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
        text_instructions, data_values, file_data_offset = parse_assembly(filename)

        # Update data_offset if specified in the file
        if file_data_offset != DEFAULT_DATA_OFFSET:
            self.data_offset = file_data_offset
        print(f"Data offset set to: 0x{self.data_offset:x}")

        # Load .text section
        for i, instruction in enumerate(text_instructions):
            encoded = encode_instruction(instruction)
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

        print(f"Executing instruction at PC {self.pc}: {instruction:032b}")
        print(f"Opcode: {opcode:07b}, rd: {rd}, rs1: {rs1}, rs2: {rs2}, funct3: {funct3:03b}, imm: {imm}")

        if opcode == 0x13:  # ADDI
            self.registers[rd] = self.registers[rs1] + imm
            print(f"ADDI: x{rd} = x{rs1} + {imm}")
        elif opcode == 0x33:  # ADD, SUB, AND, OR
            if funct3 == 0x0:
                if (instruction >> 25) == 0x20:  # SUB
                    self.registers[rd] = self.registers[rs1] - self.registers[rs2]
                    print(f"SUB: x{rd} = x{rs1} - x{rs2}")
                else:  # ADD
                    self.registers[rd] = self.registers[rs1] + self.registers[rs2]
                    print(f"ADD: x{rd} = x{rs1} + x{rs2}")
            elif funct3 == 0x7:  # AND
                self.registers[rd] = self.registers[rs1] & self.registers[rs2]
                print(f"AND: x{rd} = x{rs1} & x{rs2}")
            elif funct3 == 0x6:  # OR
                self.registers[rd] = self.registers[rs1] | self.registers[rs2]
                print(f"OR: x{rd} = x{rs1} | x{rs2}")
        elif opcode == 0x23:  # SW
            imm = ((instruction >> 25) << 5) | ((instruction >> 7) & 0x1F)
            addr = self.registers[rs1] + imm
            value = self.registers[rs2]
            print(f"SW: Storing value {value} (0x{value:08x}) to address {addr} (0x{addr:08x})")
            print(f"SW: Base address: {self.registers[rs1]}, Immediate: {imm}")
            self.memory[addr:addr+4] = value.to_bytes(4, byteorder='little')
            print(f"SW: Memory at {addr} (0x{addr:08x}) is now: {self.memory[addr:addr+4].hex()}")
        elif opcode == 0x03:  # LW
            imm = instruction >> 20
            addr = self.registers[rs1] + imm
            print(f"LW: Loading from address {addr} (0x{addr:08x})")
            print(f"LW: Base address: {self.registers[rs1]}, Immediate: {imm}")
            value_bytes = self.memory[addr:addr+4]
            self.registers[rd] = int.from_bytes(value_bytes, byteorder='little')
            print(f"LW: Loaded bytes: {value_bytes.hex()}, Interpreted as: {self.registers[rd]}")
        elif opcode == 0x73 and funct3 == 0x0:  # ecall, ebreak
            if (instruction >> 20) & 0xFFF == 0x001:  # ebreak
                print("EBREAK instruction encountered. Terminating execution.")
                return True  # Indicating termination signal

        self.pc += 4
        return False  # Continue execution

    def run(self):
        instruction_count = 0
        max_instructions = 10000  # Safeguard against infinite loops
        try:
            while self.pc < len(self.memory) and instruction_count < max_instructions:
                self.debug_output()
                if self.execute_instruction():
                    print(f"EBREAK encountered. Terminating after {instruction_count} instructions.")
                    break  # Exit if ebreak is encountered
                instruction_count += 1
                if self.pc >= len(self.memory):
                    print(f"End of memory reached. Terminating after {instruction_count} instructions.")
                    break
            if instruction_count >= max_instructions:
                print(f"Maximum instruction limit ({max_instructions}) reached. Possible infinite loop.")
        except Exception as e:
            print(f"Error during execution: {str(e)}")
        finally:
            print(f"Simulation completed. Total instructions executed: {instruction_count}")

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
