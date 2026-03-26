import socket
import json
import subprocess

HOST = "0.0.0.0"
PORT = 9000

def handle_client(conn):
    data = conn.recv(4096)
    params = json.loads(data.decode())

    cmd = ["python3", "-u", "run_local.py"]

    if params.get("t"):
        cmd.append("-t")

    if params.get("e") is not None:
        cmd.extend(["-e", str(params["e"])])

    if params.get("l"):
        cmd.extend(["-l", params["l"]])

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    # wyświetlanie
    for line in process.stdout:
        message = json.dumps({
            "type": "stdout",
            "data": line
        }) + "\n"

        conn.sendall(message.encode())

    process.wait()

    # kod wyjściowy
    end_msg = json.dumps({
        "type": "end",
        "returncode": process.returncode
    }) + "\n"

    conn.sendall(end_msg.encode())


with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()

    while True:
        conn, addr = s.accept()
        with conn:
            handle_client(conn)
