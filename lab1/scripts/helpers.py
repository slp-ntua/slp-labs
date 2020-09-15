import os
import subprocess


def run_cmd(command):
    """Run given command and return stdout as a string"""
    command = f'{os.getenv("SHELL")} -c "{command}"'
    pipe = subprocess.Popen(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )

    stdout = "".join([line.decode("utf-8") for line in iter(pipe.stdout.readline, b"")])
    pipe.stdout.close()
    returncode = pipe.wait()

    if returncode != 0:
        print(f"Warning: command {command} exited with return code {returncode}")

    return stdout


def read_wiki_txt(fname):
    pairs = []
    with open(fname, "r") as fd:
        pairs = [ln.strip().split("\t") for ln in fd.readlines()]

    return pairs
