import snap7


def get_start_offset(valve, num_valves):
    if valve < 1 or valve > num_valves:
        print("Incorrect valve provided.")
        return None

    if valve <= num_valves / 3:
        return 0
    elif valve <= (num_valves / 3) * 2:
        return 1
    else:
        return 2


def get_bit_offset(valve, num_valves):
    if valve < 1 or valve > num_valves:
        print("Incorrect valve provided.")
        return None

    return (valve - 1) % (num_valves // 3) + 1


def write_bool(valve, value, num_valves):
    start_offset = get_start_offset(valve, num_valves)
    bit_offset = get_bit_offset(valve, num_valves)

    reading = plc.db_read(DB_NUMBER, start_offset, 1)  # (db number, start offset, read 1 byte)
    snap7.util.set_bool(reading, 0, bit_offset, value)  # (value 1 = True; 0 = False)
    # (bytearray_: bytearray, byte_index: int, bool_index: int, value: bool)
    plc.db_write(DB_NUMBER, start_offset, reading)  # Write the byte array. The boolean should change


def write_values(valves, value, num_valves):
    for num in valves:
        write_bool(num, value, num_valves)
