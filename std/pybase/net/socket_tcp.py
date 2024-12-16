import socket
import time
import keyboard

class Socket_():
    def __init__(self) -> None:
        self.socket_ = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_start = True

    def server(self):
        self.server_start = True
        host = socket.gethostname()
        port = 12345  
        self.socket_.bind((host, port))
        self.socket_.listen(5)

        while self.server_start:
            client_socket, addr = self.socket_.accept()
            data = client_socket.recv(1024)  # 接收数据，最多1024字节
            print(f"sever收到消息: {data.decode('utf-8')}")
            print("server连接地址: ", addr)

            message = "欢迎访问服务器！"
            client_socket.send(message.encode('utf-8'))
            client_socket.close()
        self.socket_.close()

    def stop_server(self):
        self.server_start = False
        

def client():
    socket_ = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host = socket.gethostname()
    port = 12345

    socket_.connect((host, port))
    message = "Hello, Server!"
    print(f"client发送消息: {message}")
    socket_.sendall(message.encode('utf-8'))

    msg = socket_.recv(1024)
    print('client收到消息',msg.decode('utf-8'))

    socket_.close()

if __name__=='__main__':
    skt = Socket_()
    keyboard.add_hotkey('c', client)
    skt.server()
    # time.sleep(400)q
    # client()

