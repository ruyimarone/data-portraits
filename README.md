# Data Portraits

<img width="363" alt="image" src="https://github.com/ruyimarone/data-portraits/assets/10734779/3951d2ee-2560-4fd2-90f2-ec840f2dfbee">

<img width="450" alt="image" src="https://github.com/ruyimarone/data-portraits/assets/10734779/f3fec35c-9879-46b0-a4aa-e264dd06bf01">


This is the code for [Data Portraits: Recording Foundation Model Training Data](https://dataportraits.org/) by Marc Marone and Ben Van Durme.

Large models are trained on increasingly immense and opaque datasets, but it can be very difficult to answer a fundamental question: **Was this in a model's training set?**

We call for documentation artifacts that can answer this membership question and term these artifacts **Data Portraits.**

This repo implements one tool that can answer this question -- based on efficient hash storage with Bloom filters.
Of course, many other dataset documentation tools exist.
See our paper for details about this method, other tools, and properties that make ours unique.

For more details, see [our paper](https://openreview.net/pdf?id=ZrNRBmOzwE).

# Installing
1. Run the `install_redis.sh` script in the root of this repo. If all goes well, redis and redis bloom will be installed local to this repo.
2. Install requirements `pip install -r requirements.txt`
3. [Optional] Install this as a package if you need to import it elsewhere: `pip install -e .`

> [!NOTE]  
> If there are issues with redis or the redis installation, see the [expected structure page here](redis.md)

# Running
Try running `python easy_redis.py --just-start`. If all goes well, this will start a redis server with default parameters (check `python easy_redis.py --help`).
If this fails, check logs in `instances/`

All of this can be handled with the typical `redis-cli` interface if you are familiar with that method. 

## Loading and Using Bloom Filters

Files can be loaded manually (see `from_file` and `to_file`) but the library is now compatible with the Huggingface hub! 

```python
# after running python easy_redis.py --just-start
import dataportraits
# this downloads ~26GB. But this is much smaller than the whole dataset!
portrait = dataportraits.from_hub("mmarone/portraits-sketch-stack.50-50.bf", verbose=True)

text = """
Test sentence about Data Portraits - NOT IN THE STACK!
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")
inputs = tokenizer("Hello world!", return_tensors="pt")
outputs = model(**inputs)
"""
report = portrait.contains_from_text([text], sort_chains_by_length=True)
print(report[0]['chains'][0])
#['s import AutoTokenizer, AutoModel\ntokenizer = Auto', 'Tokenizer.from_pretrained("bert-base-uncased")\nmod', 'el = AutoModel.from_pretrained("bert-base-uncased"', ')\ninputs = tokenizer("Hello world!", return_tensor']
```

Please see our paper for details about membership testing. In particular, note the boundary and striding strategy means that not every ngram is stored - but we store enough ngrams that we can still infer whether a long sequence was part of a dataset. 

## Citing
If you find this repo or our web demo useful, please cite [our paper](https://openreview.net/pdf?id=ZrNRBmOzwE).
```
@inproceedings{
    marone2023dataportraits,
    title={Data Portraits: Recording Foundation Model Training Data},
    author={Marc Marone and Benjamin {Van Durme}},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
    year={2023},
    url={https://proceedings.neurips.cc/paper_files/paper/2023/file/3112ee706d21d734c15532c1239773e1-Paper-Datasets_and_Benchmarks.pdf}
}
```

