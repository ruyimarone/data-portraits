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


