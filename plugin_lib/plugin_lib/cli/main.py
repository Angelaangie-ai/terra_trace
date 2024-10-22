import logging

from .parser import CliParser


def main():
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] - %(asctime)s - %(message)s")
    parser = CliParser()
    args = parser.parse_args()
    parser.dispatch(args)


if __name__ == "__main__":
    main()
