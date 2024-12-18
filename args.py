import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--whitelist", type=str, default="8093,8094", help="Comma-separated list of whitelisted ports")
    args = parser.parse_args()
    args.whitelist = [int(port) for port in args.whitelist.split(',')]
    return args

args = get_args()

print(args.whitelist[0])