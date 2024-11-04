# QUIP

QUIP (**QU**oted **I**nformation **P**recision) score is our metric introduced in [According-To](https://arxiv.org/abs/2305.13252).
Informally, it measures how much some text content quotes from another corpus. QUIP is always computed with respect to a source corpus. 
More specifically, it is the portion of ngrams in an input text that appear in another corpus as measured by a membership test indicator function. 

<img width="560" alt="image" src="https://github.com/user-attachments/assets/9b42c888-84f6-4113-be57-6abd55c61723">

(see the According-To paper for more details)

One implementation of the membership test is a Data Portrait Bloom filter; other implementations are possible! Note that these Bloom filters are different from the ones used for corpus documentation (i.e. the systems described in the original Data Portraits paper).

**Crucially, QUIP systems store all ngrams, while the Portraits intended for dataset documentation store every n-th ngram (see the original Data Portraits paper for details).**

QUIP systems also use a shorter ngram length (e.g. `25` characters) as we found this worked better when building a metric. In our code base, these are referred to as width and stride. A typical QUIP system will have width 25 and stride 1. The original Portraits systems have width 50 and stride 50. 

# Running a QUIP system

Follow the installation and redis startup instructions in the [README](README.md). Then (after redis has been started!) load a system like this:

```python
import dataportraits
sketch = dataportraits.from_hub("mmarone/quip_20200301.25-1.bf", verbose=True)
sketch.contains_pretty("Testcase for data portraits as follows: A Bloom filter is a space-efficient probabilistic data structure. END PORTRAITS TEST CASE")
```
Which should output:
```
'Testcase for data portraits as follows: [A Bloom filter is a space-efficient probabilistic data structure]. END PORTRAITS TEST CASE'
```

The bracketed text indicates quoted spans. `contains_pretty` is a convenience method and is only appropriate for QUIP (i.e. stride=1) systems. For full query data, you'll need to run the server:

# QUIP Server

Starting the server:
```bash
DS_SERVER_PORT=8001 DS_NAME="mmarone/quip_20200301.25-1.bf" DS_WIDTH=25 python server.py
```

Making a query (you would typically use python requests or your other favorite HTTP library instead of wget):

```bash
#!/bin/bash

# Define the server address
SERVER_URL="localhost:8001/quip"

# JSON payload to send
read -r -d '' PAYLOAD << EOM
{
  "format_quotes": true,
  "documents": [
    "A Bloom filter is a space-efficient probabilistic data structure, conceived by Burton Howard Bloom in 1970, that is used to test whether an element is a member of a set. False positive matches are possible, but false negatives are not",
    ".tes eht ni stnemele fo rebmun ro ezis eht fo tnednepedni ,ytilibaborp evitisop eslaf %1 a rof deriuqer era tnemele rep stib 01 naht rewef ,yllareneg eroM",
    "Testcase for data portraits as follows: A Bloom filter is a space-efficient probabilistic data structure. END PORTRAITS TEST CASE",
    "The ninth floor of the Miami-Dade pretrial detention facility is dubbed the \"forgotten floor.\""
  ]
}
EOM

# Make the POST request
curl -X POST "$SERVER_URL" \
  -H "Content-Type: application/json" \
  -d "$PAYLOAD" | \
  jq 'map(del(.segments, .is_member, .chains, .chain_idxs, .quote_latex))'
```

Using `jq` to filter some of the extra information to make the output more readable (remove the `jq` pipe if you don't have `jq` on your system) should output:

```
[
  {
    "badness": 25,
    "doc": "A Bloom filter is a space-efficient probabilistic data structure, conceived by Burton Howard Bloom in 1970, that is used to test whether an element is a member of a set. False positive matches are possible, but false negatives are not",
    "expected_longest": 8.4,
    "idx": 0,
    "longest_chain": 210,
    "quip_report": {
      "denominator": 210,
      "numerator": 210,
      "quip_25_beta": 1,
      "too_short": false
    },
    "quote_text": "[A Bloom filter is a space-efficient probabilistic data structure, conceived by Burton Howard Bloom in 1970, that is used to test whether an element is a member of a set. False positive matches are possible, but false negatives are not]",
    "sketch_signature": "src/datasketches-dev/instances/2024-10-13_18-25-32_9748@localhost:8899 [quip_20200301.25-1.bf] [12693299532 elements] [21.76 GiB] ENV: localhost 8899 quip_20200301.25-1.bf 25"
  },
  {
    "badness": 0,
    "doc": ".tes eht ni stnemele fo rebmun ro ezis eht fo tnednepedni ,ytilibaborp evitisop eslaf %1 a rof deriuqer era tnemele rep stib 01 naht rewef ,yllareneg eroM",
    "expected_longest": 5.2,
    "idx": 1,
    "longest_chain": 0,
    "quip_report": {
      "denominator": 130,
      "numerator": 0,
      "quip_25_beta": 0,
      "too_short": false
    },
    "quote_text": ".tes eht ni stnemele fo rebmun ro ezis eht fo tnednepedni ,ytilibaborp evitisop eslaf %1 a rof deriuqer era tnemele rep stib 01 naht rewef ,yllareneg eroM",
    "sketch_signature": "src/datasketches-dev/instances/2024-10-13_18-25-32_9748@localhost:8899 [quip_20200301.25-1.bf] [12693299532 elements] [21.76 GiB] ENV: localhost 8899 quip_20200301.25-1.bf 25"
  },
  {
    "badness": 9.523809523809524,
    "doc": "Testcase for data portraits as follows: A Bloom filter is a space-efficient probabilistic data structure. END PORTRAITS TEST CASE",
    "expected_longest": 4.2,
    "idx": 2,
    "longest_chain": 40,
    "quip_report": {
      "denominator": 105,
      "numerator": 40,
      "quip_25_beta": 0.38095238095238093,
      "too_short": false
    },
    "quote_text": "Testcase for data portraits as follows: [A Bloom filter is a space-efficient probabilistic data structure]. END PORTRAITS TEST CASE",
    "sketch_signature": "src/datasketches-dev/instances/2024-10-13_18-25-32_9748@localhost:8899 [quip_20200301.25-1.bf] [12693299532 elements] [21.76 GiB] ENV: localhost 8899 quip_20200301.25-1.bf 25"
  },
  {
    "badness": 3.2142857142857144,
    "doc": "The ninth floor of the Miami-Dade pretrial detention facility is dubbed the \"forgotten floor.\"",
    "expected_longest": 2.8,
    "idx": 3,
    "longest_chain": 9,
    "quip_report": {
      "denominator": 70,
      "numerator": 11,
      "quip_25_beta": 0.15714285714285714,
      "too_short": false
    },
    "quote_text": "The ninth floor of the Miami-Dad[e pretrial detention facility][[ is ]][dubbed the \"forgotten ]floor.\"",
    "sketch_signature": "src/datasketches-dev/instances/2024-10-13_18-25-32_9748@localhost:8899 [quip_20200301.25-1.bf] [12693299532 elements] [21.76 GiB] ENV: localhost 8899 quip_20200301.25-1.bf 25"
  }
]
```

