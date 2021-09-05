from datetime import datetime


def timestamp(format="%Y-%m-%dT%H:%M:%S"):
    return datetime.now().strftime(format)
