import socket
import threading
import pickle
import io
import queue
import struct

host = '0.0.0.0' 
port = 9090  
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind((host, port))
server_socket.listen()
tot_len=60720

def recivstatus(c_socket,rq): #워커노드로부터 파라미터를 수신하는 함수
    while 1:
        blen=0
        tot_data=b''
        tot_len=c_socket.recv(4)
        tot_len=struct.unpack('!I', tot_len)[0]
        print(tot_len)
        while (tot_len-blen>0):
            data = c_socket.recv(tot_len)
            blen+=len(data)
            tot_data+=data
        result =  pickle.Unpickler(io.BytesIO(tot_data)).load()
        rq.put(result)

def mean(rq):#수신 받은 파라미터를 키맞춰서 평균 연산
    f=rq.get()
    s=rq.get()
    result = f.copy()
    for key in f.copy().keys():
        result[key] = (f[key] + s[key]) / 2.0
    return result

def sendstatus(group, rq):#워커노드 두개의 파라미터가 전부 준비되면 워커노드에게 파라미터 전송
    while 1:
        if rq.qsize()==2:
            print('s')
            mean_dict=mean(rq)
            dict_byte=pickle.dumps(mean_dict)
            dict_byte=bytearray(dict_byte)
            for i in group:
                i.sendall(struct.pack('!I', len(dict_byte)))
                i.sendall(dict_byte)

rq=queue.Queue(2)
flag=queue.Queue(2)
group=[]

while 1:
    c_socket, addr = server_socket.accept()#워커노드 접속
    group.append(c_socket)#워커노드 큐를 구성한다
    print('conn')
    print('connected client addr:', addr)
    tre=threading.Thread(target=recivstatus, args=(c_socket,rq,))#수신부분 스레드화
    tre.start()
    tse=threading.Thread(target=sendstatus, args=(group, rq,))#송신부분 스레드화
    tse.start()

server_socket.close()
