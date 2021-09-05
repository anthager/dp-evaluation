#!/usr/bin/python3.7
# -*- coding: utf-8 -*-


def log(arg, text):
    status = {
        "info": style("[INFO]", "green"),
        "debug": style("[DBUG]", "blue"),
        "error": style("[ERRR]", "red"),
        "warn": style("[WARN]", "yellow")
    }[arg]

    print(status, text)


def style(output, color, styles=[]):
    if color is not None:
        try:
            output = {
                "yellow": "\033[93m%s",
                "red": "\033[91m%s",
                "green": "\033[92m%s",
                "blue": "\033[94m%s"
            }[color] % output
        except Exception:
            output = ""
    for style in styles:
        output = {
            "bold": "\033[1m%s",
        }[style] % output

    return output + "\033[0m"  # default


def log_elapsed_time(start_time, end_time):
    hours, r = divmod(end_time - start_time, 3600)
    minutes, seconds = divmod(r, 60)
    log("info", "Elapsed time: %dh%dm%ds" % (hours, minutes, seconds))


def log_test_params(synthetic_test, memory_test, meta, num_of_synthetic_datasets):
    log("info", "Dataset '%s' loaded" % meta.name)
    log("debug", "Number of tests: %s" % meta.num_of_runs)
    log("debug", "Dataset sizes: %s" % str(meta.dataset_sizes)[1:-1])
    log("debug", "Epsilons: %s" % str(meta.epsilons)[1:-1])
    if synthetic_test:
        log("info", "Running synthetic test, loading %s synthetic datasets..." %
            num_of_synthetic_datasets)
    elif memory_test:
        log("info", "Running memory test...")
