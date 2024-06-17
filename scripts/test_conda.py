import msgpack
import os


def main():
    print('Works')
    # print current conda environment
    print(msgpack.version)
    print('Current conda environment:')
    print(os.environ)


if __name__ == '__main__':
    main()
