import snap7
import numpy as np

# Variables
NUMBER_OF_VALVES = 24
DATABASE_VALUES = 1


def get_start_offset(valve, num_valves):
    if valve < 0 or valve >= num_valves:
        print("Incorrect valve provided.")
        return None

    if valve + 1 <= num_valves / 3:
        return 0
    elif valve + 1 <= (num_valves / 3) * 2:
        return 1
    else:
        return 2


def get_bit_offset(valve, num_valves):
    if valve < 0 or valve >= num_valves:
        print("Incorrect valve provided.")
        return None

    bt_off = (valve) % (num_valves // 3)
    return bt_off + 1 if bt_off % 2 == 0 else bt_off - 1


def write_bool(plc, valve, value):
    start_offset = get_start_offset(valve, NUMBER_OF_VALVES)
    bit_offset = get_bit_offset(valve, NUMBER_OF_VALVES)

    reading = plc.db_read(DATABASE_VALUES, start_offset, 1)  # (db number, start offset, read 1 byte)
    snap7.util.set_bool(reading, 0, bit_offset, value)  # (value 1 = True; 0 = False)
    # (bytearray_: bytearray, byte_index: int, bool_index: int, value: bool)
    plc.db_write(DATABASE_VALUES, start_offset, reading)  # Write the byte array. The boolean should change


def write_values(plc, valves, value):
    for num in valves:
        write_bool(plc, num, value)


if __name__ == '__main__':
    plc = snap7.client.Client()
    plc.connect('192.168.0.1', 0, 1)  # IP address, rack, slot (from HW settings)
    write_values(plc, np.array([9, 10, 13, 14]), 0)
