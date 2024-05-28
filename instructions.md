# Installing
1. Run the `install_redis.sh` script in the root of this repo. If all goes well, redis and redis bloom will be installed local to this repo.
2. Install requirements `pip install -r requirements.txt`
3. [Optional] Install this as a package if you need to import it elsewhere: `pip install -e .`

# Running
Try running `python easy_redis.py --just-start`. If all goes well, this will start a redis server with default parameters (check `python easy_redis.py --help`).
If this fails, check logs in `instances/`

All of this can be handled with the typical `redis-cli` interface if you are familiar with that method. 

## Loading Bloom Filters

> [!WARNING]  
> Need to download model files somehow

Load a sketch with `python easy_redis.py --start-from-dir /path/to/bf/directory` 

## Use loaded filters

Now you can do:
```python
import dataportraits

# localhost:8899 is the default for the redis server started above
# wikipedia.50-50.bf is the name of the system - see the easy_redis.py script for more
# change as necessary!
portrait = dataportraits.RedisBFSketch('localhost', 8899, 'wiki-demo.50-50.bf', 50)

text = """
Test sentence about Data Portraits - NOT IN WIKIPEDIA!
Bloom proposed the technique for applications where the amount of source data would require an impractically large amount of memory if "conventional" error-free hashing techniques were applied
"""
report = portrait.contains_from_text([text])
print(report[0]['chains'])

# [['cations where the amount of source data would requ', 'ire an impractically large amount of memory if "co', 'nventional" error-free hashing techniques were app']]

```

## Shutting down 
Note that you'll neeed to shutdown the server later, otherwise it can consume large amounts of ram:
```shell
python easy_redis.py --shutdown-redis
```
**This will delete the bloom filter in memory - meaniing any modified sketches will be deleted**

To prevent this, save the state of the redis server first:
```shell
python easy_redis.py --dump-to-dir /directory/to/save/filters/
```
