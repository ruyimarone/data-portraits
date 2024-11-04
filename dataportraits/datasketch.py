import redis
import numpy as np
import struct
import math
import tqdm
import sys
import json
import itertools
import contextlib
import os
import random
import huggingface_hub
from collections import namedtuple
from functools import partial

import dataportraits.utils as utils
import dataportraits.code_proc as code_proc
from dataportraits import span_utils

BFInfo = namedtuple("BFInfo", "chain_size chain_n_filters options growth bytes bits current_size error bits_per_element hashes max_entries n2")
BF_INFO_STRUCT = "=QLLLQQQddLQB"

def check_chain(membership_tests, index, step, accumulator):
    if index >= len(membership_tests):
        return accumulator

    is_member = membership_tests[index][0]
    if is_member:
        accumulator.append(index)
        return check_chain(membership_tests, index + step, step, accumulator)

    return accumulator

# def infer_memberships(memberships, stride=25):
    # new_memberships = list(memberships) # copy the base

def chain_overlaps(membership_tests, width):
    already_found = set()
    matches = []
    idxs = []
    for ix, (_, _) in enumerate(membership_tests):
        if ix in already_found:
            continue

        run_idxs = check_chain(membership_tests, ix, step=width, accumulator=[])

        if len(run_idxs) == 0:
            continue

        run_segments =  [membership_tests[i][1] for i in run_idxs]
        matches.append(run_segments)
        idxs.append(run_idxs)
        already_found.update(run_idxs)

    return matches, idxs


def build_text_pipeline_fn(**kwargs):
    tokenizer_string = kwargs.get('tokenizer', None)
    use_numpy = kwargs.get('use_numpy', False)
    apply_code_processor = kwargs.get('apply_code_processor', False)

    if tokenizer_string:
        tokenizer = utils.init_tokenizer(tokenizer_string)
        from transformers.utils import logging
        logging.set_verbosity(40)

        if use_numpy:
            tokenizer_fn = partial(tokenizer, return_tensors='np', padding=True, return_attention_mask=False)
            def post_proc_fn(chunk):
                return chunk.tobytes()
        else:
            tokenizer_fn = partial(tokenizer, padding=False, return_attention_mask=False)
            def post_proc_fn(chunk):
                return chunk

        pad_token_id = tokenizer.pad_token_id
    else:
        # mock the tokenizer object
        def tokenizer_fn(batches_of_text, **kwargs):
            tokenizer_fn.pad_token_id = None
            return {'input_ids' : batches_of_text}

        def post_proc_fn(chunk):
            return chunk

        pad_token_id = None

    # after this point, a tokenizer always exists, even though it might just be a noop

    if apply_code_processor:
        def pre_process(text):
            return code_proc.proc_code(text)
    else:
        print("[WARNING] not using code proc", file=sys.stderr)
        def pre_process(text):
            return text

    # get parameters
    stride = kwargs.get('stride')
    width = kwargs.get('width')

    def pipeline(batches_of_text):
        batches_of_text = [pre_process(text) for text in batches_of_text]
        tokens = tokenizer_fn(batches_of_text)['input_ids']

        results = []
        for instance in tokens:
            ngrams = utils.chunk_sequence_strided(instance, width, stride, stop_token=pad_token_id)
            post_processed = [post_proc_fn(ngram) for ngram in ngrams]
            results.append(post_processed)
        return results
    return pipeline

def expected_strided_matches(sequence, width):
    """expected_strided_matches.
    Returns the number of expected matches if `sequence` was embedded exactly once in a corpus.
    Accounts for strided n-grams and alignment issues, but doesn't account for document boundaries or false positives.

    :param sequence:
    :param width:
    """
    if len(sequence) < width:
        return 0
    return (len(sequence) - (width - 1)) / width

class RedisBFSketch:

    def __init__(self, host, port, key, width):
        self.host = host
        self.port = port
        self.redis_client = redis.Redis(host=host, port=port)
        self.bf_client = self.redis_client.bf()
        self.key = key
        self.query_batch_size = 10000
        self.tokenizer = None
        self.width = width
        assert type(width) == int
        self.self_test()
        self.text_pipeline = build_text_pipeline_fn(width=self.width, stride=1, apply_code_processor=True) #TODO expose this as a command line

    def set_tokenizer(self, tokenizer_string):
        self.tokenizer = tokenizer_string

    def chunk(self, batch_of_strings):
        return utils.flatten_batched(self.text_pipeline(batches_of_text=batch_of_strings))

    def tokenize_and_chunk(self, batch_of_strings):
        text_pipeline = build_text_pipeline_fn(tokenizer=self.tokenizer, use_numpy=self.tokenizer is not None, width=self.width, stride=1)
        return utils.flatten_batched(text_pipeline(batches_of_text=batch_of_strings))


    def contains_from_text(self, documents, stride=0, sort_chains_by_length=False, pretty=False, infer_by_stride=False):

        if pretty:
            raise NotImplementedError("todo")

        if self.tokenizer:
            lens, segments = self.tokenize_and_chunk(documents)
        else:
            lens, segments = self.chunk(documents)
        import dataportraits.timers as timers
        # with timers.Timer("Contains All"):
        results = self.contains_all(segments)

        membership_results = utils.unflatten(lens, zip(results, segments), empty_element = [[], []])
        assert len(membership_results) == len(documents), "Didn't get membership results for some document"

        outputs = []
        for doc_num, (original_document, segment_memberships) in enumerate(zip(documents, membership_results)):

            # hack
            original_document = code_proc.proc_code(original_document)

            doc_report = {
                'idx' : doc_num,
                'doc' : original_document,
                'segments' : [],
                'is_member' : [],
                'chains' : [],
                'chain_idxs' : [],
                'longest_chain' : 0,
                'expected_longest' : 0,
                'badness' : 0.0
            }
                # 'n_runs' : 0

            if self.tokenizer:
                raise NotImplementedError("todo")


            if len(segment_memberships[0]) > 0:
                membership_tests, segments = zip(*segment_memberships) # unzip

                # if infer_by_stride:
                    # membership_tests = infer_memberships(membership_tests, stride=self.width)

                segment_memberships = list(zip(membership_tests, segments)) # rezip, hacky

                # if we have stride, use it
                if stride != 0:
                    doc_report['chains'], doc_report['chain_idxs'] = chain_overlaps(segment_memberships, stride)
                # otherwise assume it is the same as width
                else:
                    doc_report['chains'], doc_report['chain_idxs'] = chain_overlaps(segment_memberships, self.width)

                doc_report['segments'] = segments
                doc_report['is_member'] = [bool(i) for i in membership_tests]
                if sort_chains_by_length:
                    assert all((len(a) == len(b)) for a, b in zip(doc_report['chains'], doc_report['chain_idxs']))
                    doc_report['chains'] = sorted(doc_report['chains'], key = lambda x : len(x), reverse=True) # these are sorted the same way because each element has the same length. asserted above.
                    doc_report['chain_idxs'] = sorted(doc_report['chain_idxs'], key = lambda x : len(x), reverse=True)

                # set badness
                if len(doc_report['chain_idxs']) == 0:
                    doc_report['longest_chain'] = 0
                else:
                    doc_report['longest_chain'] = max(len(chain) for chain in doc_report['chain_idxs'])
                # doc_report['expected_longest'] = math.ceil(expected_strided_matches(original_document, self.width))
                doc_report['expected_longest'] = expected_strided_matches(original_document, self.width)

                doc_report['badness'] = doc_report['longest_chain'] / doc_report['expected_longest']

            outputs.append(doc_report)

        return outputs
    
    def contains_pretty(self, text):
        report = self.contains_from_text([text], stride=1, sort_chains_by_length=True)[0]
        report['quote_text'] = span_utils.format_chains(report, lambda num_quotes : "[" * num_quotes, lambda num_quotes : "]" * num_quotes)
        return report['quote_text']

    def add_all(self, ngrams_to_add):
        # This is very slow due to overhead
        # with sending and parsing redis responses. 
        # A faster method involves redis in pipe mode, but this has added complexity. 
        # This method is sufficient for relatively small datasets (i.e. not full LLM corpora)
        # hiredis (pip install redis[hiredis]) also helps
        self.bf_client.madd(self.key, *ngrams_to_add)
        return

    def contains(self, item):
        return self.contains_all([item])[0]

    def contains_all(self, items):
        self.exists()
        results = []
        for batch in utils.batcher_fn(items, self.query_batch_size):
            results.extend(self.bf_client.mexists(self.key, *batch))
        return results

    def self_test(self):
        self.redis_client.ping()

    def exists(self):
        assert self.redis_client.exists(self.key) == 1, f"Key `{self.key}` doesn't exist in the specified server"

    def stats(self):
        self.exists()
        info_bytes = self._scandump(self.key, 0)[1]
        return BFInfo._make(struct.unpack(BF_INFO_STRUCT, info_bytes))

    def _scandump(self, key, iter):
        #monkey patch to bypass hiredis
        #https://github.com/redis/redis-py/blob/936d49f4c1dd6cf0c2e3ad80de29f25eef81d8a9/redis/commands/bf/commands.py
        params = [key, iter]
        options = {}
        options[redis.client.NEVER_DECODE] = []
        return self.redis_client.execute_command(redis.commands.bf.BF_SCANDUMP, *params, **options)


    def iter_dump(self, verbose=True, return_iter=False):
        #dump the raw bytes. https://redis.io/commands/bf.scandump/

        with utils.get_progress(unit_scale=True, unit_divisor = 1024, unit='iB', total=self.stats().bytes) if verbose else contextlib.nullcontext() as progress:
            iter = 0
            while True:
                iter, data = self._scandump(self.key, iter)
                if iter == 0:
                    return
                else:
                    if verbose:
                        progress.update(len(data))
                    if return_iter:
                        yield iter, data
                    else:
                        yield data

    def to_file(self, file_path, verbose=False, legacy=False):
        # file_path  is a path like /some/destination/name.bf
        # i.e. a file not a directory
        # in non legacy mode, this is the index file
        # it should end with .bf

        assert file_path.endswith(".bf")

        if not legacy:
            max_bytes = 2 * 1024 ** 3 # 2 GiB
            idxs = []
            part_num = 0
            current_size = 0
            part_path = f"{file_path}.part{part_num}.bin"

            f = open(part_path, 'wb')  # Open first part file

            for it, (iter, block) in enumerate(self.iter_dump(return_iter=True, verbose=verbose)):
                block_size = len(block)
                if current_size + block_size > max_bytes:
                    f.close()
                    part_num += 1
                    part_path = f"{file_path}.part{part_num}.bin"
                    f = open(part_path, 'wb')  # Open new part file
                    current_size = 0

                idx = {
                    'iter': iter, # iter is the redis internal cursor, needed to load the parts back in
                    'block_num': it,
                    'block_size': block_size,
                    'part_file': os.path.basename(part_path) # local to the destination
                }
                idxs.append(idx)
                f.write(block)
                current_size += block_size

            f.close()  # Close the last part file

            metadata = {
                'blocks'  : idxs,
                'bf_info' : self.stats()._asdict(),
                'width'   : self.width,
                'key'     : self.key,
                'last-loaded' : {'host' : self.host, 'port' : self.port}
            }

            with open(file_path, 'w') as f:
                json.dump(metadata, f, indent=2)

        else:
            idxs = []
            with open(file_path, 'wb') as f:
                for it, (iter, block) in enumerate(self.iter_dump(return_iter=True, verbose=verbose)):
                    idx = {'iter' : iter, 'block_num' : it, 'block_size' : len(block)}
                    idxs.append(idx)
                    f.write(block)

            if file_path == '/dev/null': #TODO is this just for tests?
                return

            with open(file_path + '.idx', 'w') as f:
                json.dump(idxs, f, indent=2)

    def to_hub(self, hub_path, temp_directory, overwrite=False, verbose=False):
        assert os.path.isdir(temp_directory)

        api = huggingface_hub.HfApi()

        # write to temp file path in a safe temp directory
        temp_file = os.path.join(temp_directory, self.key)
        self.to_file(temp_file, verbose=verbose, legacy=False)

        api.create_repo(repo_id=hub_path, private=True, exist_ok=overwrite)
        api.upload_folder(repo_id=hub_path, folder_path=temp_directory)


    @classmethod
    def from_hub(cls, hf_hub_path, host="localhost", port="8899",
            key=None, overwrite=False, verbose=False):

        path = huggingface_hub.snapshot_download(hf_hub_path)
        bf_files = [f for f in os.listdir(path) if f.endswith('.bf')]
        assert len(bf_files) == 1, "Multiple .bf files were found, hf repo might be broken"
        bf_path_name = bf_files[0]
        bf_full_path = os.path.join(path, bf_path_name)

        if verbose:
            print(path, file=sys.stderr)

        with open(bf_full_path, 'r') as f:
            metadata = json.load(f)

        if key is None:
            key = metadata['key']

        return cls.from_file(host=host,
                port=port,
                key = key,
                width = metadata['width'],
                path = bf_full_path,
                overwrite = overwrite,
                verbose = verbose,
                legacy = False)

    @classmethod
    def from_file(cls, host, port, key, width, path, overwrite=False, verbose=False, legacy=False):
        bf = cls(host, port, key, width)

        if verbose:
            progress = utils.get_progress(unit_scale=True, unit_divisor = 1024, unit='iB')

        if not legacy:

            with open(path) as f:
                metadata = json.load(f)

            idxs = metadata['blocks']

            if bf.redis_client.exists(key):
                if overwrite:
                    bf.redis_client.delete(key)
                else:
                    raise Exception(f"Redis instance already contains key: {key}")

            current_part = None
            f = None

            for block in idxs:
                part_name = block['part_file']
                redis_cursor = block['iter']
                block_size = block['block_size']
                part_path = os.path.join(os.path.dirname(path), part_name)
                if current_part != part_path:
                    if f:
                        f.close()
                    f = open(part_path, 'rb')
                    current_part = part_path

                data = f.read(block_size)
                bf.bf_client.loadchunk(key, redis_cursor, data)
                if verbose:
                    progress.update(block_size)

            assert width == metadata['width'], f"Width in from_file was {width} but metadata said it should be {metadata['width']}"
            # set the metadata back
            bf.width = metadata['width']
            print(metadata['bf_info'])
            print(bf.stats())

        else:
            with open(path + '.idx') as f:
                idxs = json.load(f)


            if bf.redis_client.exists(key):
                if overwrite:
                    bf.redis_client.delete(key)
                else:
                    raise Exception(f"Redis instance already contains key: {key}")

            with open(path, 'rb') as f:
                for idx in idxs:
                    # print(idx['block_num'])
                    data = f.read(idx['block_size'])
                    bf.bf_client.loadchunk(key, idx['iter'], data)
                    if verbose:
                        progress.update(idx['block_size'])

        return bf

    def checksum(self, nth_block=1):
        size_in_bytes = self.stats().bytes
        block_size = 16777216 # 16 MiB
        # approximate because blocks are not certain to be this size
        # it's just an estimate of how many blocks redis will return
        approx_num_blocks = size_in_bytes // block_size
        approx_num_blocks_to_sample = 1 + approx_num_blocks // nth_block # also count the 0th block

        if approx_num_blocks_to_sample <= 10:
            nth_block = 1
            print("Bloom Filter is too small to sample from, checksum all blocks instead", file=sys.stderr)

        hashes = []
        _, blocks = self._splitdump()
        for n, block in enumerate(blocks):
            if n % nth_block == 0:
                hashes.append(hash(block))

        print(f"Hashed {len(hashes)} blocks out of {n+1} total blocks", file=sys.stderr)

        assert len(hashes) > 0, "Didn't get enough blocks to checksum"

        return hash(tuple(hashes))


    def count_bits(self):
        dump = self.iter_dump()
        _ = BFInfo._make(struct.unpack(BF_INFO_STRUCT, next(dump)))
        set_bits = 0
        for block in dump:
            set_bits += utils.sum_bits_from_packed(np.frombuffer(block, dtype=np.uint8))
        return set_bits

    def _splitdump(self):
        dump = self.iter_dump(verbose=True, return_iter=False)
        header = BFInfo._make(struct.unpack(BF_INFO_STRUCT, next(dump)))
        return header, dump

    def approximate_size(self):
        # Bloom Filters are initialized with an estimated number of elements
        # You can also work backwards and use the load factor (number of set bits)
        # to approximate the number of inserted items
        bf_info = self.stats()
        num_bits = bf_info.bits
        num_hashes = bf_info.hashes

        set_bits = self.count_bits() # this can take a long time
        p_1 = set_bits / num_bits # probability that a bit is 1

        return int(-(num_bits / num_hashes) * math.log(1 - p_1))

    def __repr__(self):
        stats = self.stats()
        # 1024 based! GB = 2 ** 30
        size_in_gb = stats.bytes / (1024 ** 3)
        dir = self.redis_client.config_get('dir')['dir']
        return f"{dir}@{self.host}:{self.port} [{self.key}] [{stats.current_size} elements] [{size_in_gb:.2f} GiB]"

if __name__ == '__main__':
    text = "The scientist named the population, after their distinctive horn, Ovid’s Unicorn. These four-horned, silver-white unicorns were previously unknown to science."
    sketch = RedisBFSketch('localhost', 8899, 'pile_04.str.20', 20)
    results = sketch.contains_from_text([text, "too short"], sort_chains_by_length=True)
    from pprint import pprint
    for r in results:
        pprint(r)
