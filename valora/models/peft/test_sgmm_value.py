import torch
R = 256
N = 32*1024
H = 4096
x = torch.randn((N, H), dtype=torch.float16, device="cuda")
delta_qA = torch.zeros((len(x), R), dtype=torch.float16, device="cuda")
delta_oA = torch.zeros((len(x), H), dtype=torch.float16, device="cuda")
num_layers = 1
# for sgmm
dtype = torch.float16
device = torch.device("cuda")
x_list = [x for _ in range(num_layers)]
x_ptr_l = [t.data_ptr() for t in x_list]
x_ptr = torch.tensor(x_ptr_l, dtype=torch.int64, device=device)
y_list = [delta_qA for _ in range(num_layers)]
y_ptr_l = [t.data_ptr() for t in y_list]
y_ptr = torch.tensor(y_ptr_l, dtype=torch.int64, device=device)
o_list = [delta_oA for _ in range(num_layers)]
o_ptr_l = [t.data_ptr() for t in o_list]
o_ptr = torch.tensor(o_ptr_l, dtype=torch.int64, device=device)
s_list = [64 for _ in range(num_layers)]
s_0 = torch.tensor(s_list, dtype=torch.int32, device=device)
s_list = [64 for _ in range(num_layers)]
s_1 = torch.tensor(s_list, dtype=torch.int32, device=device)

key_buffer = torch.randn((R, H), dtype=torch.float16, device="cuda")
value_buffer = torch.randn((H, R), dtype=torch.float16, device="cuda")
w_list = [key_buffer for _ in range(num_layers)]
w_ptr_l = [t.data_ptr() for t in w_list]
k_ptr = torch.tensor(w_ptr_l, dtype=torch.int64, device=device)
w_list = [value_buffer for _ in range(num_layers)]
w_ptr_l = [t.data_ptr() for t in w_list]
v_ptr = torch.tensor(w_ptr_l, dtype=torch.int64, device=device)
type_slice = torch.randn((2,2), dtype=dtype, device=device)