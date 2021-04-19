import os


def get_relative_path_from_root(filename: str):
    return os.sep.join([__file__[:-17].replace('/', os.sep), 'datasets', filename])


def get_freq(seconds: int):
    minutes = seconds / 60
    hours = minutes / 60
    days = hours / 24
    freq = 'D'
    if days >= 31:
        freq = 'MS'
    elif 355 <= days:
        freq = str(int(days / 31)) + 'M'
    elif days > 365:
        freq = str(int(days / 365)) + 'Y'
    return freq
