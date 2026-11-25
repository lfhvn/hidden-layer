"""Run mlx_lm.server and restart it whenever its generation thread dies.

    python3 serve.py --model <path> --port 8080 [other mlx_lm.server args]

On long runs the server sometimes hits "[metal::malloc] Resource limit
(499000) exceeded": its generation thread dies, but the process stays up and
answers every request with 404. This wrapper watches the log for that and
restarts the server; the harness retries failed requests long enough to ride
out the restart (about a minute).
"""

import subprocess
import sys
import time

DEATH = "generation thread died"
QUIET = ('" 200 -', '" 404 -')


def main():
    cmd = ["mlx_lm.server", *sys.argv[1:]]
    restarts = 0
    while True:
        print(f"[serve] starting (restart {restarts}): {' '.join(cmd)}", flush=True)
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        died = False
        for line in proc.stdout:
            if not any(q in line for q in QUIET):
                sys.stdout.write(line)
                sys.stdout.flush()
            if DEATH in line and not died:
                died = True
                print(f"[serve] {time.strftime('%H:%M:%S')} generation thread died; restarting", flush=True)
                proc.terminate()
                try:
                    proc.wait(30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                break
        proc.wait()
        restarts += 1
        if not died:
            print(f"[serve] server exited with code {proc.returncode}; restarting in 5s", flush=True)
        time.sleep(5)


if __name__ == "__main__":
    main()
