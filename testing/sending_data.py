import serial
import time

esp = serial.Serial('COM9', 115200, timeout=2)
time.sleep(2) 

user_input = int(input("Open (1) or closed (0)? "))

if user_input == 1:
    esp.write(b'OPEN\n')
else:
    esp.write(b'CLOSED\n')

print(esp.readline().decode().strip())

esp.close()