import itertools

def per_token_map(report, width=25):
    document = report['doc']
    counts = [0] * len(document)
    for chain in report['chain_idxs']:
        start_idx = min(chain)
        end_idx = max(chain) + width
        assert end_idx <= len(document) # end will be exclusive, so it can be equal to len(document)
        for idx in range(start_idx, end_idx):
            counts[idx] += 1
    return counts, document

def split_array(arr):
    return [list(group) for _, group in itertools.groupby(arr)]

def default_start_policy(value):
    return "["

def default_end_policy(value):
    return "]"

def latex_hl_start(value):
    return "{{\\sethlcolor{{quoted!{}}}\\hl{{".format(value * 20)

def latex_hl_end(value):
    return "}}"

def format_chains(report, start=default_start_policy, end=default_end_policy):
    counts, doc = per_token_map(report)
    edges = split_array(counts)

    parts = []
    doc_iter = iter(doc)
    for group in edges:
        parts.append(start(group[0]))
        parts.append(itertools.islice(doc_iter, len(group)))
        parts.append(end(group[-1]))

    return ''.join(itertools.chain.from_iterable(parts))
