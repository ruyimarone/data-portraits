If you're familiar with redis already and want to use an existing server or a system installation,
you can use that databse. Just make sure you know your redis connection details and have RedisBloom loaded into 
your server (with config files or command line args)

Included launch scripts assume this repo structure but it's easily changed:

```
├── dataportraits
│   ├── datasketch.py
│   ├── __init__.py
├── easy_redis.py
├── RedisBloom
│   └── bin/linux-x64-release/redisbloom.so
├── redis_configs
├── redis-stable
│   └── bin/redis-server
└── start_redis_ez.sh
```

For clarity, we write out `install_redis.sh` here. Note the submodules and specific version tag.

```shell
set -ex

# base redis
wget https://download.redis.io/redis-stable.tar.gz
tar -xzf redis-stable.tar.gz
cd redis-stable
make -j 4
make install PREFIX="$(pwd)" # install to redis-stable/bin
cd ..

# Install bloom filter package`
git clone https://github.com/RedisBloom/RedisBloom.git
cd RedisBloom
git checkout tags/v2.4.3
git submodule update --init --recursive
make -j 4
cd ..
```

Our code was tested against RedisBloom version 2.4.3. *We assume a certain binary header structure in serialized Bloom filter files, other redis versions may change this and break the file format!*