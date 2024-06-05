import socket
import time
import keyboard

class Socket_():
    def __init__(self) -> None:
        self.socket_ = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def server(self):

        host = socket.gethostname()
        port = 12345  

        self.socket_.bind((host, port))

        self.socket_.listen(5)

        while True:
            client_socket, addr = self.socket_.accept()
            print("连接地址: ", addr)

            message = "欢迎访问服务器！"
            client_socket.send(message.encode('utf-8'))

            client_socket.close()

def client():
    socket_ = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host = socket.gethostname()
    port = 12345  

    socket_.connect((host, port))

    msg = socket_.recv(1024)

    print(msg.decode('utf-8'))

    socket_.close()

if __name__=='__main__':
    skt = Socket_()
    keyboard.add_hotkey('q', client)
    skt.server()
    # time.sleep(400)q
    # client()

