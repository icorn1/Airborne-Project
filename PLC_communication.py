import snap7
import numpy as np


def get_start_offset(valve, num_valves):
    if valve < 0 or valve > num_valves:
        print("Incorrect valve provided.")
        return None

    if valve + 1<= num_valves / 3:
        return 0
    elif valve + 1<= (num_valves / 3) * 2:
        return 1
    else:
        return 2


def get_bit_offset(valve, num_valves):
    if valve < 0 or valve > num_valves:
        print("Incorrect valve provided.")
        return None
    
    bt_off = (valve) % (num_valves // 3)
    return bt_off + 1 if bt_off % 2 == 0 else bt_off - 1  


def write_bool(plc, valve, value, num_valves):
    start_offset = get_start_offset(valve, num_valves)
    bit_offset = get_bit_offset(valve, num_valves)

    reading = plc.db_read(1, start_offset, 1)  # (db number, start offset, read 1 byte)
    snap7.util.set_bool(reading, 0, bit_offset, value)  # (value 1 = True; 0 = False)
    # (bytearray_: bytearray, byte_index: int, bool_index: int, value: bool)
    plc.db_write(1, start_offset, reading)  # Write the byte array. The boolean should change


def write_values(plc, valves, value, num_valves):
    for num in valves:
        write_bool(plc, num, value, num_valves)


if __name__ == '__main__':
    plc = snap7.client.Client()
    plc.connect('192.168.0.1', 0, 1)  # IP address, rack, slot (from HW settings)
    write_values(plc, np.array([9, 10, 13, 14]), 0, 24)
    

