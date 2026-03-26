from prefect import flow, task
from prefect import get_run_logger
import time


# ---------------------------
# Mock processors
# ---------------------------

@task
def load_image():
    time.sleep(0.5)
    return "image"


@task
def preprocess(x):
    logger = get_run_logger()
    logger.info("running preprocess")
    time.sleep(0.5)
    return f"preprocessed({x})"


@task
def segment(x):
    time.sleep(0.5)
    return f"segmented({x})"


@task
def analyze(x):
    time.sleep(0.5)
    return f"analysis({x})"


@task
def classify(x):
    time.sleep(0.5)
    return f"classified({x})"


@task
def coincidence(a, b):
    time.sleep(0.5)
    return f"coincidence({a}, {b})"


# ---------------------------
# DAG 1: Chain
# ---------------------------

@flow(name="chain_flow")
def chain_flow():
    x = load_image.submit()
    x = preprocess.submit(x)
    x = segment.submit(x)
    x = analyze.submit(x)
    return x


# ---------------------------
# DAG 2: Branch
# ---------------------------

@flow(name="branch_flow")
def branch_flow():
    x = load_image.submit()
    x = preprocess.submit(x)
    x = segment.submit(x)

    a = analyze.submit(x)
    b = classify.submit(x)

    return a, b


# ---------------------------
# DAG 3: Merge
# ---------------------------

@flow(name="merge_flow")
def merge_flow():
    ch1 = preprocess.submit("channel_1")
    ch2 = preprocess.submit("channel_2")

    seg1 = segment.submit(ch1)
    seg2 = segment.submit(ch2)

    result = coincidence.submit(seg1, seg2)
    return result


# ---------------------------
# Run all + keep UI alive
# ---------------------------

if __name__ == "__main__":
    print("\n--- Running Chain Flow ---")
    print(chain_flow())

    print("\n--- Running Branch Flow ---")
    print(branch_flow())

    print("\n--- Running Merge Flow ---")
    print(merge_flow())

    print("\nServer running — open Prefect UI now")
    time.sleep(120)