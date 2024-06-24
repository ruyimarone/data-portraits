import sys
import time

def num_to_bytes(some_int):
    return str(some_int).encode()

def generate_redis_protocol_basic(*cmd):
    """generate_redis_protocol_basic. Returns a single redis command byte string. 

    :param cmd:
    """

    arg_byte = b"$"
    array_byte = b"*"
    empty_byte = b""
    line_end_bytes = b"\r\n"
    proto = empty_byte.join((array_byte, num_to_bytes(len(cmd)), line_end_bytes))
    # linear builder pattern

    argument_buffer = [proto]
    for elt in cmd:
        if type(elt) is not bytes:
            elt = elt.encode()
        argument_buffer.append(empty_byte.join((arg_byte, num_to_bytes(len(elt)), line_end_bytes, elt, line_end_bytes)))
    proto = empty_byte.join(argument_buffer)

    return proto

def print_b(some_bytes):
    sys.stdout.buffer.write(some_bytes)

if __name__ == '__main__':

    N = 1000000
    xs = [str(i) for i in range(N)]

    start = time.time()
    for x in xs:
        print_b(generate_redis_protocol_basic("PFADD", "hll.bench", x))
        # print_b(generate_redis_buffered("PFADD", "hll.bench", x))
    end = time.time()

    print((end - start) / N, file=sys.stderr)
    print(end - start, file=sys.stderr)
