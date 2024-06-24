import argparse
import os
import sys
import redis
import dataportraits
import subprocess

start_script = './start_redis_ez.sh'

def is_redis_running(host, port):
    redis_client = redis.Redis(host=host, port=port)
    return redis_client.ping()

def start_redis(args):
    redis_uri = f"{args.host}:{args.port}"
    try: # see if redis is already running and accessible
        redis_client = redis.Redis(host=args.host, port=args.port)
        redis_client.ping()
        print(f"redis was already running at {redis_uri}", file=sys.stderr)
        return # if it's running we're done
    except: # redis is not running
        if args.host != 'localhost':
            # remote machine doesn't have redis running and we can't start it there so crash
            raise Exception(f"Redis was not found on the remote machine: {args.host}")
        else:
            # redis is not running on localhost, but it should be
            # this blocks
            print(f"trying to start redis at {redis_uri}", file=sys.stderr)
            proc = subprocess.run(f"bash {start_script} {args.port}", shell=True, check=True, capture_output=True)
            try:
                redis_client = redis.Redis(host=args.host, port=args.port)
                redis_client.ping()
                print(f"started redis at {redis_uri}", file=sys.stderr)
                return # we started and can exit
            except:
                print(proc.stderr.decode('utf-8'), file=sys.stderr)

                raise Exception("Failed to start redis")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    command_group = parser.add_mutually_exclusive_group(required=True)
    command_group.add_argument("--start-from-dir", type=str, help="Start redis, load all bloom filters from a directory", default=None)
    command_group.add_argument("--start-from-hub", type=str, help="Start redis, load one bloom filter from a hub path", default=None)
    command_group.add_argument("--just-start", action='store_true', help="Start redis and don't do anything else")
    command_group.add_argument("--dump-to-dir", type=str, help="Dump all filters from a redis instance to a directory", default=None)
    command_group.add_argument("--shutdown-redis", action='store_true', help="Shutdown redis instance")

    parser.add_argument("--host", type=str, help="redis host", default='localhost')
    parser.add_argument("--port", type=int, help="redis port", default=8899)
    parser.add_argument("--allow-key-overwrites", action='store_true', help="Allow overwriting the redis keys")
    parser.add_argument("--no-create-dirs", action='store_true', help="Don't create the directory to write to")
    parser.add_argument("--migrate-temp-dir", type=str, help="Use with start-from-dir, open bfs in legacy mode, and upload them to the HF hub")
    parser.add_argument("--hf-prefix", type=str, help="Name to use in huggingface e.g. mmarone")

    args = parser.parse_args()

    redis_uri = f"{args.host}:{args.port}"

    if args.dump_to_dir:


        if not is_redis_running(args.host, args.port):
            raise Exception(f"Redis is not running at {redis_uri}, can't dump anything")

        redis_client = redis.Redis(host=args.host, port=args.port)

        if not args.no_create_dirs:
            os.makedirs(args.dump_to_dir, exist_ok=True)

        assert os.path.isdir(args.dump_to_dir)

        keys = redis_client.keys('*.bf')
        clean_keys = []
        final_paths = []
        for key in keys:
            key = key.decode('utf-8')
            try:
                w, s = key.split('.')[-2].split('-') # try to split `some.complex.name.width-stride.bf` into width and stride
                w = int(w)
                s = int(s)
            except:
                raise Exception(f"Couldn't parse {key}, aborting without any writes")
            clean_keys.append((key, w, s))
            path = os.path.join(args.dump_to_dir, key)
            assert not os.path.exists(path), f"Path {path} already exists, aborting"
            final_paths.append(path)

        for path, (key, width, stride) in zip(final_paths, clean_keys):
            sketch = dataportraits.RedisBFSketch(args.host, args.port, key, width)
            print("writing:", sketch)
            print("to:", path)
            sketch.to_file(path, verbose=True)

        sys.exit(0)

    if args.start_from_dir or args.just_start or args.start_from_hub:
        start_redis(args)

        if args.just_start:
            print("Only starting was requested", file=sys.stderr)
            sys.exit(0)

        if args.start_from_dir:
            assert os.path.isdir(args.start_from_dir)
            if args.migrate_temp_dir:
                os.makedirs(args.migrate_temp_dir, exist_ok=True)
                # assert os.path.isdir(args.migrate_temp_dir), "Need a temp directory to use for migration"
                assert args.hf_prefix, "Need a name to upload to"
                import huggingface_hub
                api = huggingface_hub.HfApi()

            filters = []
            for filename in os.listdir(args.start_from_dir):
                path = os.path.join(args.start_from_dir, filename)
                if path.endswith('.bf'):
                    try:
                        w, s = filename.split('.')[-2].split('-') # try to split `some.complex.name.width-stride.bf` into width and stride
                        w = int(w)
                        s = int(s)
                    except:
                        raise Exception(f"Couldn't parse {key}, aborting without loading any filters")
                    # path, key, width, stride
                    filters.append((path, filename, w, s))

            for path, key, width, stride in filters:
                print(f"Try to load {key} from {path} to {redis_uri}...")
                sketch = dataportraits.RedisBFSketch.from_file(args.host, args.port, key, width, path, overwrite=args.allow_key_overwrites, legacy=True, verbose=True)
                print(f"Successfully loaded {sketch}")
                if args.migrate_temp_dir:

                    # make a sub directory for this model (otherwise we will upload everything)
                    temp_dir_for_this_specific_sketch = os.path.join(args.migrate_temp_dir, key)
                    os.makedirs(temp_dir_for_this_specific_sketch, exist_ok=True)

                    repo_id = f"{args.hf_prefix}/portrait-{key}"
                    print(f"Write to {temp_dir_for_this_specific_sketch} and upload to {repo_id}")
                    sketch.to_hub(repo_id, temp_dir_for_this_specific_sketch, verbose=True)

        elif args.start_from_hub:
            s = dataportraits.from_hub(args.start_from_hub, host=args.host, port=args.port, overwrite=args.allow_key_overwrites, verbose=True)
            print(s)
        else:
            raise Exception("Bad args?")


    if args.shutdown_redis:
        redis_client = redis.Redis(host=args.host, port=args.port)
        redis_client.shutdown()
