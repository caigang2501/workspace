import socket
import time
import keyboard

class Socket_():
    def __init__(self) -> None:
        self.socket_ = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def server(self):
        host = socket.gethostname()
        port = 12345
        self.socket_.bind((host, port))

        while self.server_start:
            data, addr = self.socket_.recvfrom(1024)
            print(f"serverc收到来自 {addr} 的消息: {data.decode('utf-8')}")

            response = "服务器已收到消息"
            self.socket_.sendto(response.encode('utf-8'), addr)
        self.socket_.close()
        
def client():
    socket_ = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    host = socket.gethostname()
    server_address = (host, 12345)

    # socket_.connect((host, port))
    message = "Hello, UDP Server!"
    print(f"client发送消息: {message}")
    socket_.sendto(message.encode('utf-8'),server_address)

    data, server = socket_.recvfrom(1024)
    print(f"client收到服务器响应: {data.decode('utf-8')}")

    socket_.close()

if __name__=='__main__':
    skt = Socket_()
    keyboard.add_hotkey('c', client)
    skt.server()
    # time.sleep(400)q
    # client()

