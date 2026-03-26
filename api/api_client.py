import socket
import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-t", action="store_true", help="trening")
    parser.add_argument("-e", type=int, help="ile epochów")
    parser.add_argument("-l", type=str, help="ścieżka pliku")

    return parser.parse_args()


def build_payload(args):
    payload = {}

    if args.t:
        payload["t"] = True

    if args.e is not None:
        payload["e"] = args.e

    if args.l:
        payload["l"] = args.l

    return payload


def main():
    args = parse_args()
    payload = build_payload(args)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect(("localhost", 9000))

        s.sendall((json.dumps(payload) + "\n").encode())

        buffer = ""

        while True:
            chunk = s.recv(1024).decode()
            if not chunk:
                break

            buffer += chunk

            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                msg = json.loads(line)

                if msg["type"] == "stdout":
                    print(msg["data"], end="")

                elif msg["type"] == "end":
                    print(f"\nExit code: {msg['returncode']}")
                    return


if __name__ == "__main__":
    main()
