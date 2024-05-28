port=${1:-8899}

portraits_root="$(pwd)"
base_dir="./instances"
datestamp=$(date +"%Y-%m-%d_%H-%M-%S")
run_name="${datestamp}"_"$RANDOM"
re_bloom="RedisBloom/bin/linux-x64-release/redisbloom.so"
conf_file="./redis_configs/redis.conf"

#echo $run_name
run_dir=$base_dir/$run_name
#echo $run_dir

# copy config and start redis
mkdir -p $run_dir
cp $conf_file $run_dir
cd $run_dir

$portraits_root/redis-stable/bin/redis-server ./redis.conf --loadmodule $portraits_root/$re_bloom --port $port --daemonize yes
sleep 2 # give redis a chance to start
$portraits_root/redis-stable/bin/redis-cli -p $port "ping"

if [ $? -eq 0 ]; then
    echo $HOSTNAME:$port
    exit 0
else
    echo "Failed to start, check log in $run_dir"
    exit 1
fi


