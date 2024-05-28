from dataportraits.utils import batcher_fn
from dataportraits import RedisBFSketch
import sys
import torch

def test_batcher():
    data = [[1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]]

    expected = [
            [[1, 2 , 3], [4,5,6]],
            [[7, 8, 9]]
        ]

    for ix, batch in enumerate(batcher_fn(data, 2)):
        assert batch == expected[ix]


    # make sure we can slice torch arrays (and get out a torch array)
    # i.e. the data type is preserved and slices are fast
    data = torch.randn(20, 5)

    for batch in batcher_fn(data, 8):
        assert isinstance(batch, torch.Tensor)

    # make sure it has a lazy path for stream
    for batch in batcher_fn((row for row in data), 8):
        assert isinstance(batch, list) #constructing from a stream, so return a list of rows
        assert isinstance(batch[0], torch.Tensor) # each item in the list is a tensor

def test_chunk_ngrams():
    from dataportraits.utils import chunk_sequence_strided

    # basic tests
    seq = [1, 2, 3, 4, 5, 6, 7]

    assert list(chunk_sequence_strided(seq, 3)) == [[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7]]
    assert list(chunk_sequence_strided(seq, 10)) == []

    seq = [1, 10, 100, 1000]
    assert list(chunk_sequence_strided(seq, 3)) == [[1, 10, 100], [10, 100, 1000]]

    # padding tests
    seq = [1, 2, 3, 100, 100, 100]
    assert list(chunk_sequence_strided(seq, 2, stop_token=100)) == [[1, 2], [2, 3]]


    # stride tests
    seq = [10, 11, 12, 13, 14, 15]
    assert list(chunk_sequence_strided(seq, 4, 4)) == [[10, 11, 12, 13]]
    assert list(chunk_sequence_strided(seq, 2, 2)) == [[10, 11], [12, 13], [14, 15]]
    assert list(chunk_sequence_strided(seq, 2, 20)) == [[10, 11]]
    assert list(chunk_sequence_strided(seq, 4, 2)) == [[10, 11, 12, 13], [12, 13, 14, 15]]

    seq = [10, 11, 12, 13, 14, 15, 100, 100, 100]
    assert list(chunk_sequence_strided(seq, 4, stride = 2, stop_token = 100)) == [[10, 11, 12, 13], [12, 13, 14, 15]]

def test_redis_proto():
    from dataportraits.redis_protocol import generate_redis_protocol_basic

    assert generate_redis_protocol_basic("A", "BB", "CCC") == b'*3\r\n$1\r\nA\r\n$2\r\nBB\r\n$3\r\nCCC\r\n'

def test_chains():
    from dataportraits.utils import flatten_batched, unflatten

    basic = [[9], [1, 2, 3], [4, 5, 6, 7]]
    lens, elts = flatten_batched(basic)
    assert lens == [1, 3, 4]
    assert elts == [9, 1, 2, 3, 4, 5, 6, 7]
    reconstructed = unflatten(lens, elts)
    assert reconstructed == basic

    lens, elts = flatten_batched([])
    assert lens == []
    assert elts == []
    reconstructed = unflatten(lens, elts)
    assert reconstructed == []

    lens, elts = flatten_batched([[], []])
    assert lens == [0, 0]
    assert elts == []
    reconstructed = unflatten(lens, elts)
    assert reconstructed == [[], []]


    lens, elts = flatten_batched([[], [1, 2], []])
    assert lens == [0, 2, 0]
    assert elts == [1, 2]
    reconstructed = unflatten(lens, elts)
    assert reconstructed == [[], [1, 2], []]

    # test that it works even if the things to be unflattened are iterators, not lists
    lens, elts = flatten_batched([(n for n in range(0)), (n for n in range(4))])
    assert lens == [0, 4]
    assert elts == list(range(4))
    reconstructed = unflatten(lens, elts)
    assert reconstructed == [[], [0, 1, 2, 3]]
    reconstructed_with_extra = unflatten(lens, zip(elts, range(len(elts))))
    assert reconstructed_with_extra == [[], [(0, 0), (1, 1), (2, 2), (3, 3)]]

def test_sketch_preproc():
    import numpy as np
    from dataportraits.utils import DUMMY_TOKENIZER_NAME 

    # mock the redis connection
    class RedisMock:
        def __init__(self, *args, **kwargs):
            print("!!! Using a MOCK REDIS !!!", file=sys.stderr)
            self.bf = lambda *args : None
            self.ping = lambda *args : True

    import redis
    redis.Redis = RedisMock
    sketch = RedisBFSketch(None, None, None, width=6)
    sketch.exists = lambda *args : None

    # end mocks, start tests
    docs = ["test document", "another"]
    lens, flat_batch = sketch.chunk(docs)
    assert flat_batch == ["test d", "est do", "st doc", "t docu",
            " docum", "docume", "ocumen", "cument", "anothe", "nother"]
    assert lens == [8, 2]

    sketch.set_tokenizer(DUMMY_TOKENIZER_NAME) # uses a dummy tokenizer
    sketch.width = 4
    docs = ["This is a test document .", "Another"]
    lens, flat_batch = sketch.tokenize_and_chunk(docs)
    # convert back to lists, from numpy byte buffers
    flat_batch = [np.frombuffer(b, dtype=np.int64).tolist() for b in flat_batch]
    assert lens == [3, 0] # the second doc is too short
    assert flat_batch == [[1459, 599, 2119, 3776],
                            [599, 2119, 3776, 3323],
                            [2119, 3776, 3323, 2398]]


def test_feeder_preproc():
    from feeder import init_worker, worker_fn
    from types import SimpleNamespace

    mock_args = SimpleNamespace()
    mock_args.tokenizer = None
    mock_args.width  = 10
    mock_args.stride = 10
    mock_args.command = 'FAKE_COMMAND'
    mock_args.key = 'FAKE_KEY'
    mock_args.multiple = True

    init_worker(mock_args)
    results = worker_fn(["This is a test document .", "Another"])
    assert len(results) == 1
    results = results[0]
    line_end_bytes = b"\r\n"
    parts = [b.decode() for b in results.split(line_end_bytes)]

    # should generate 2 ngrams: 'This is a ' and 'test docum' of length 10
    # 'Another' is too short, less than width ( = 10)
    assert parts == ['*4', '$12', 'FAKE_COMMAND', '$8', 'FAKE_KEY', '$10', 'This is a ', '$10', 'test docum', '']

# options for text pipeline
# can have tokenizer
# can be a mock tokenizer
