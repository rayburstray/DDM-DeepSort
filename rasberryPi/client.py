import socket
import keyboard

def main():
    with socket.socket(socket.AF_INET,
                        socket.SOCK_STREAM) as s:
        s.connect(('10.52.61.126', 12345))
        try:
            print("Press 'w', 'a', 's', 'd' to control. Press 'q' to quit.")
            while True:
                if keyboard.is_pressed('w'):
                    s.sendall('w'.encode())
                    response = s.recv(1024).decode()
                    print(f"Server response: {response}")
                elif keyboard.is_pressed('a'):
                    s.sendall('a'.encode())
                    response = s.recv(1024).decode()
                    print(f"Server response: {response}")
                elif keyboard.is_pressed('s'):
                    s.sendall('s'.encode())
                    response = s.recv(1024).decode()
                    print(f"Server response: {response}")
                elif keyboard.is_pressed('d'):
                    s.sendall('d'.encode())
                    response = s.recv(1024).decode()
                    print(f"Server response: {response}")
                elif keyboard.is_pressed('q'):
                    s.sendall('q'.encode())
                    s.close()
                    print("Exiting.")
                    break
        except KeyboardInterrupt:
            print("Exiting.")

if __name__ == "__main__":
    main()
