import socket
import threading
import pickle
import json

def recvall(sock, n):
    # Helper function to recv n bytes or return None if EOF is hit
    fragments = []
    size = 0

    while n > size:
        chunk = sock.recv(n - size)
        if not chunk:
            return None
        fragments.append(chunk)
        size += len(chunk)

    return b''.join(fragments)


def recv_msg(sock):
    # Read message length and unpack it into an integer
    raw_msglen = recvall(sock, 4)
    if raw_msglen is None:
        return None

    msglen = int.from_bytes(raw_msglen, "little")
    raw_msg = recvall(sock, msglen)
    return raw_msg



def listen_conn(conn, message_target, threaded=True, repeat=True):
    error = False
    result = ""
    while True:
        my_message = recv_msg(conn)
        if my_message is None:
            error = True
            return result, error

        if threaded:
            new_thread = threading.Thread(target=message_target, args=[my_message])
            new_thread.start()

        else:
            result = message_target(my_message)

        if not repeat:
            return result, error


def send_message(msg, sock=None, host=None, port=None):
    if sock is None:
        if host is None or port is None: raise KeyError("Missing target to connect to")

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((host, port))

    msg_raw = pickle.dumps(msg)

    msg = len(msg_raw).to_bytes(4, "little") + msg_raw
    sock.sendall(msg)

def send_json(json_to_send:dict, sock):
    our_message = json.dumps(json_to_send)
    our_message = our_message.encode("utf-8")
    msg = len(our_message).to_bytes(4, "big") + our_message

    sock.sendall(msg)


def process_msg_as_json_string(msg):
    string_data = msg.decode('utf-8')
    result = json.loads(string_data)
    return result