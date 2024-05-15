import numpy as np
import pickle, struct, socket, math
import numpy as np
import pickle, struct, socket, math
import torch
import sys
import time
import torchvision
import random
import numpy as np
import math
import copy

def send_msg(sock, msg):
    msg_pickle = pickle.dumps(msg)
    sock.sendall(struct.pack(">I", len(msg_pickle)))
    sock.sendall(msg_pickle)
   # print(msg[0], 'sent to', sock.getpeername())


def recv_msg(sock, expect_msg_type=None, map_location=None):
    # msg_len = struct.unpack(">I", sock.recv(4))[0]
    # msg = sock.recv(msg_len, socket.MSG_WAITALL)
    # msg = pickle.loads(msg)
    # print(msg[0], 'received from', sock.getpeername())
    #
    # if (expect_msg_type is not None) and (msg[0] != expect_msg_type):
    #     raise Exception("Expected " + expect_msg_type + " but received " + msg[0])
    # return msg
    # Read exactly 4 bytes for the message length
    msg_len_data = b''
    while len(msg_len_data) < 4:
        packet = sock.recv(4 - len(msg_len_data))
        if not packet:
            raise ConnectionError("Socket connection closed prematurely while reading length")
        msg_len_data += packet
    msg_len = struct.unpack(">I", msg_len_data)[0]

    # Read the message data based on the length received
    msg_data = b''
    while len(msg_data) < msg_len:
        packet = sock.recv(msg_len - len(msg_data))
        if not packet:
            raise ConnectionError("Socket connection closed prematurely during message receipt")
        msg_data += packet

    import io
    use_cuda = torch.cuda.is_available()
    if map_location is None:
        map_location = 'cpu' if not use_cuda else None

    # Deserialize the message
    try:
        buffer = io.BytesIO(msg_data)
        msg = torch.load(buffer, map_location=map_location)
    except (RuntimeError, pickle.UnpicklingError) as e:
        if isinstance(e, RuntimeError) and 'CUDA' in str(e):
            buffer.seek(0)
            msg = torch.load(buffer, map_location='cpu')
        else:
            msg = pickle.loads(msg_data)


    # Deserialize the message
    #msg = pickle.loads(msg_data)
    print(msg[0], 'received from', sock.getpeername())

    # Check if the received message type matches the expected type
    if expect_msg_type is not None and msg[0] != expect_msg_type:
        raise Exception("Expected " + expect_msg_type + " but received " + msg[0])

    return msg

def partition_way_converse(partition):
    for i in range(len(partition)):
        if partition[i]==0:
            partition[i]=1
        else:
            partition[i]=0
    return partition

def time_count(start_time, end_time):
    durasec = int(end_time - start_time)
    duramsec = int((end_time - start_time - int(end_time - start_time)) * 1000)
    return float(durasec*1000+duramsec)
    
def printer(content):
    print(content)
    #fid = "/Users/brladder77/result/VGG_emnist_hfl_coopfl_fedmec_random.txt"
    #fid = "/Users/brladder77/result/AlexNet_cifar10_hfl_coopfl_fedmec_random(local_manual).txt"
    fid = "/home/doyoung/Coopfl-demo-main/result/AlexNet_cifar10_hfl_coopfl_fedmec_random_sls.txt"
    with open(fid,'a') as fid:
        content = content.rstrip('\n') + '\n'
        fid.write(content)
        fid.flush()

def printer_model(content):
    print(content)
    fid = "/Users/brladder77/result/20201107.txt"
    with open(fid,'a') as fid:
        fid.write(str(content))
        fid.flush()


def time_printer(start_time, end_time, model,i,forward):
    durasec = int(start_time - end_time)
    duramsec = int((start_time - end_time - int(start_time - end_time)) * 1000)
    durammsec = int(((start_time - end_time - durasec)*1000 - duramsec)*1000)
    if forward==1:
        printer("Forward, Layer:{}-{} output type:{} size:{:.2f}MB,runtime:{}s{}ms{}us".format(i-1,i,model.shape,
            sys.getsizeof(model.storage()) / (1024 * 1024), durasec, duramsec,durammsec))
    else:
        printer("Backward Layer:{}-{}  output type:{} size:{:.2f}MB,runtime:{}s{}ms{}us".format(i,i-1,model.shape, sys.getsizeof(
            model.storage()) / (1024 * 1024), durasec, duramsec,durammsec))


def add_model(dst_models, src_models):
    for (dst_model, src_model) in zip(dst_models, src_models):
        params1 = src_model.named_parameters()
        params2 = dst_model.named_parameters()
        dict_params2 = dict(params2)
        with torch.no_grad():
            for name1, param1 in params1:
                if name1 in dict_params2:
                    dict_params2[name1].set_(
                        param1.data + dict_params2[name1].data)
    return dst_models

def minus_model(dst_model, src_model):
    params1 = dst_model.state_dict().copy()
    params2 = src_model.state_dict().copy()
    with torch.no_grad():
        for name1 in params1:
            if name1 in params2:
                params1[name1] = params1[name1] - params2[name1]
    model = copy.deepcopy(dst_model)
    model.load_state_dict(params1, strict=False)
    return model

def scale_model(models, scale):
    for model in models:
        params = model.named_parameters()
        dict_params = dict(params)
        with torch.no_grad():
            for name, param in dict_params.items():
                dict_params[name].set_(dict_params[name].data * scale)
    return models


def start_forward_layer(partition_way):
    for i in range(len(partition_way)):
        if partition_way[i]==0:
            return i
        else:
            return -1

def start_backward_layer(partition_way):
    for i in range(len(partition_way)-1,0,-1):
        if partition_way[i]==0:
            return i
        else:
            return -1

def time_duration(start_time, end_time):
    durasec = int(start_time - end_time)
    duramsec = int((start_time - end_time - int(start_time - end_time)) * 1000)
    return durasec, duramsec

