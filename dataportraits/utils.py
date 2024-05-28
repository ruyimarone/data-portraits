import sys
import numpy as np
import itertools
import tqdm

DUMMY_TOKENIZER_NAME = 'DUMMY_TOKENIZER'
def init_tokenizer(name_string):
    if name_string == DUMMY_TOKENIZER_NAME:
        return dummy_tokenizer

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(name_string)
    if 'gpt2' in name_string:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def chunk_sequence_strided(sequence, width, stride=1, stop_token = -1):

    l = len(sequence)

    if width < l:
        yield from ()

    end_ix = l - width + 1
    for ix in range(0, end_ix, stride):
        ngram_idx = ix + width - 1 # idx of the final token in the ngram
        if sequence[ngram_idx] == stop_token:
            return
        yield sequence[ix : ix + width]

# flatten a batched sequence, but keep track of which element came from which batch
# by recording how long each batch was
def flatten_batched(batched_sequence):
    lengths = []
    flattened = []
    for batch in batched_sequence:
        element_ix = 0
        for element in batch:
            flattened.append(element)
            element_ix += 1

        lengths.append(element_ix)
    return lengths, flattened

# reconstruct batches from a flattened version. replaces empty iterators with empty lists
def unflatten(lengths, elements, empty_element=[]):
    accumulator = []
    # current_idx = 0
    elements = iter(elements)
    for l in lengths:
        if l == 0:
            accumulator.append(empty_element)
            continue
        else:
            batch = list(itertools.islice(elements, l))
            assert len(batch) == l
            accumulator.append(batch)
            # current_idx += l
    return accumulator

def batcher_fn(data, batch_size):
    is_slicable = True
    try:
        _ = data[:batch_size]
    except:
        is_slicable = False

    if is_slicable:
        _ = data[:batch_size]
        ix = 0
        while True:
            batch = data[ix:ix+batch_size]
            if len(batch) == 0:
                return
            yield batch
            ix += batch_size
    else:
        buffer = []
        for item in data:
            buffer.append(item)
            if len(buffer) == batch_size:
                yield buffer
                buffer = []
        if len(buffer) > 0:
            yield buffer

def max_size_batcher(data, max_size):
    current_size = 0
    buffer = []
    for item in data:
        buffer.append(item)
        current_size += len(item)
        if current_size >= max_size:
            yield buffer
            current_size = 0
            buffer = []

    if len(buffer) > 0:
        yield buffer


# def look_ahead_batcher(data, look_ahead_size, max_size):
    # for lookahead in batcher_fn(data, look_ahead_size):
        # lookahead = sorted(lookahead, key = lambda x : len(x))
        # yield from max_size_batcher(lookahead, max_size)

def look_ahead_batcher(data, look_ahead_size, batch_size):
    for lookahead in batcher_fn(data, look_ahead_size):
        lookahead = sorted(lookahead, key = lambda x : len(x))
        yield from batcher_fn(lookahead, batch_size)

def testtt():
    import random
    synth = [("*" * random.randint(1, 100)) + '\n' for _ in range(1000)]
    q = list(look_ahead_batcher(synth, 66, 1000))
    for i in q:
        print(i)


class SanitizedWriter:
    def __init__(self, stream=sys.stderr):
        self.stream = stream
        self.control_table = dict.fromkeys(range(32))

    def write(self, string):
        # old = string
        string = string.translate(self.control_table)
        string = string.replace("\r", "").replace("\n", "").replace("[A", "") + '\n'
        if string.strip() != '':
            self.stream.write(string)

    def flush(self):
        self.stream.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)

def get_progress(*args, smoothing=0, **kwargs):
    if not sys.stderr.isatty():
        kwargs['file'] = SanitizedWriter(sys.stderr)
        kwargs['mininterval'] = 1.0
    return tqdm.tqdm(*args, smoothing=smoothing, **kwargs)

def aid(x):
      # This function returns the memory
      # block address of an array.
      return x.__array_interface__['data'][0]


def dummy_tokenizer(batch_of_text, **kwargs):
    old_state = np.random.get_state()  # store randomness state
    np.random.seed(1000) # deterministic fake tokenization

    # if 'pad_token_id' in kwargs:
    #     setattr(dummy_tokenizer, 'pad_token_id', kwargs['pad_token_id'])
    # else:
    #     dummy_tokenizer.pad_token_id = -1

    bsz = len(batch_of_text)
    lens = [len(b.split()) for b in batch_of_text]
    m = max(lens)
    base = np.full((bsz, m), dummy_tokenizer.pad_token_id, dtype=np.int64)
    
    for ix, l in enumerate(lens):
        base[ix, :l] = np.random.randint(0, 4096, size=(l))

    if 'return_tensors' in kwargs:
        # default to numpy
        pass
    else:
        base = base.tolist()

    results = {'input_ids' : base}
    np.random.set_state(old_state) # restore the old random state
    return results

dummy_tokenizer.pad_token_id = -1

def identity(*args):
    return args

sum_table = np.unpackbits(np.arange(256, dtype=np.uint8)).reshape((-1, 8)).sum(axis=1).astype(np.uint8)
def sum_bits_from_packed(packed_array):
    """sum_bits_from_packed.

    Interprets the packed uint8 array as a bit vector
    and counts (sums) the number of set bits
    using a memory efficient table lookup.

    :param packed_array: a uint8 flat numpy array
    """
    assert packed_array.dtype == np.uint8
    return np.sum(sum_table[packed_array])

# def unzip(zipped):
    # try:
        # return zip(*zipped)
    # except ValueError:
        # return [[] for 
