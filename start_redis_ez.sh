port=${1:-8899}

base_dir="./instances"
datestamp=$(date +"%Y-%m-%d_%H-%M-%S")
run_name="${datestamp}"_"$RANDOM"
re_bloom="RedisBloom/redisbloom.so"
conf_file="./redis_configs/redis.conf"

#echo $run_name
run_dir=$base_dir/$run_name
#echo $run_dir

# copy config and start redis
mkdir -p $run_dir
cp $conf_file $run_dir
cd $run_dir

redis-stable/bin/redis-server ./redis.conf --loadmodule ../../$re_bloom --port $port --daemonize yes
sleep 2 # give redis a chance to start

if [ -e "redis.pid" ]
then
    echo "pid file exists, redis was started!" 1>&2
    echo $HOSTNAME:$default_port
    exit 0
fi

echo "Failed to start, check log in $run_dir"
exit 1
