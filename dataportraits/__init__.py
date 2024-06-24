from .datasketch import RedisBFSketch


def from_hub(hf_hub_path, host="localhost", port="8899",
            key=None, overwrite=False, verbose=False):
    return RedisBFSketch.from_hub(hf_hub_path=hf_hub_path, host=host, port=port,
                                   key=key, overwrite=overwrite, verbose=verbose)

