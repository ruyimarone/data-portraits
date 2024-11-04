import dataportraits
from dataportraits.timers import Timer
import dataportraits.utils
import itertools, tqdm


from datasets import load_dataset

ds = load_dataset("abisee/cnn_dailymail", "3.0.0")

# Set parameters - width of 25 characters per ngram, stride of 1 (every ngram is considered, instead of skipping some)
text_pipeline = dataportraits.datasketch.build_text_pipeline_fn(stride=1, width=25, apply_code_processor=True)

# the filter needs to be initialized before this will work
# e.g. echo "bf.reserve 'cnn_dailymail.str.code.quip.25-1.bf' 0.001 3000000000 NONSCALING" | redis-cli -p 8899
bf = dataportraits.RedisBFSketch("localhost", 8899, "cnn_dailymail.str.code.quip.25-1.bf", 25)

doc_batch_size = 100 # batch size for processing documents into ngrams
ngram_batch_size = 100_000 # number of elements in a single redis command (e.g. BF.MADD)

# if the dataset is large, use a lazy iterable here instead of materializing all the documents into a list
with Timer("Load HF dataset"):
    doc_stream = tqdm.tqdm(ds['train']['article'])

doc_batch_stream = dataportraits.utils.batcher_fn(doc_stream, batch_size=doc_batch_size)
ngram_stream = itertools.chain.from_iterable(itertools.chain.from_iterable(map(text_pipeline, doc_batch_stream)))

with Timer("Add ngrams:"):
    for batch_of_ngrams in dataportraits.utils.batcher_fn(ngram_stream, 100_000):
        bf.add_all(batch_of_ngrams)

# optionally upload to the hf hub
# os.makedirs("./temp/", exist_ok=True)
# bf.to_hub("mmarone/cnn_dailymail.str.code.quip.25-1.bf", "./temp", overwrite=True, verbose=True)