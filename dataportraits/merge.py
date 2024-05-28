import datasketch
import os
import numpy as np
import struct
import json
import argparse


def block_to_numpy(block):
    return np.frombuffer(block, dtype=np.uint8)

def merge_headers(headers):
    header_values = []
    for header in headers:
        header = header[1] # discard the iter_stamp
        header_values.append(list(struct.unpack(datasketch.BF_INFO_STRUCT, header)))
        print(header_values[-1])

    # idxs we care about:
    # 0 (chain_size)
    # 6 (current_size)
    idxs_to_sum = [0, 6]
    new_header_values = header_values[0][:] # copy
    for idx in idxs_to_sum:
        new_value = sum(header[idx] for header in header_values)
        new_header_values[idx] = new_value

    return struct.pack(datasketch.BF_INFO_STRUCT, *new_header_values)

def or_filters(*filters):
    # headers = []
    block_streams = []
    headers = []
    verbose = False

    # first handle the header (this is the first block returned by iter_dump
    # consumes the header block - then leave everything else for merging
    for f in filters:
        s = f.iter_dump(return_iter=True, verbose=verbose)
        iter_num, header = next(s)
        headers.append((iter_num, header))
        block_streams.append(s)
        verbose = False

    new_header = merge_headers(headers)

    # iter_stamp, modified header packed back into bytes
    yield headers[0][0], new_header


    try:
        while True:
            blocks = []
            iter_stamps = []
            for stream in block_streams:
                iter_num, block = next(stream)
                blocks.append(block_to_numpy(block))
                iter_stamps.append(iter_num)
            assert all((i == iter_stamps[0] for i in iter_stamps)), "Uneven blocks"

            # now or the blocks for this iteration
            np_buffer = np.zeros_like(blocks[0])
            for block in blocks:
                np_buffer |= block
            yield iter_stamps[0], np_buffer.tobytes()

    except StopIteration:
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="Merge Bloom filters of the same size. Streams directly to disk, avoiding extra memory usage.")

    parser.add_argument('--keys', nargs='+', help="keys to merge - `dataset.50-50.bf` or `hostname:9999/dataset.50-50.bf`", type=str, required=True)
    parser.add_argument('--out', type=str, help="Output filename", required=True)

    args = parser.parse_args()

    bfs = []
    for key in args.keys:
        try:
            if '/' in key:
                connection_string, key_name = key.split('/')
            else:
                connection_string = 'localhost:8899'
                key_name = key
            host, port = connection_string.split(':')
            width, _ = key_name.split('.')[-2].split('-') # try to split `some.complex.name.width-stride.bf` into width and stride
            width = int(width)
        except:
            raise Exception(f"Couldn't parse {key}, aborting without writing")
        bf = datasketch.RedisBFSketch(host, int(port), key_name, width)
        print(bf)

        bfs.append(bf)

    assert not os.path.exists(args.out)

    with open(args.out, 'wb') as f:
        idxs = []
        for it, (iter_stamp, merged_bytes) in enumerate(or_filters(*bfs)):
            idx = {'iter' : iter_stamp, 'block_num' : it, 'block_size' : len(merged_bytes)}
            idxs.append(idx)
            f.write(merged_bytes)

        # if path == '/dev/null':
            # return

        with open(args.out + '.idx', 'w') as f:
            json.dump(idxs, f, indent=2)



