# import os
# # print(__name__)
# # print(os.getcwd())
# from transformer import Attend
# import torch


# def test_attend():
#     model = Attend(0.1, True, False)
#     q = torch.rand(8, 200, 64) # b, h, i, d
#     k = torch.rand(8, 200, 64) # b, h, i, d
#     v = torch.rand(8, 200, 64) # b, h, i, d
#     with torch.no_grad():
#         y = model(q, k, v)
#         print("y", y.shape)

# test_attend()

# def test_transformer():

#     '''
#     self,
#     *,
#     dim,
#     depth,
#     heads,
#     dim_context = None,
#     cross_attend = False,
#     attn_dropout = 0.,
#     ff_dropout = 0.,
#     grad_shrink_alpha = 0.1,
#     cond_as_self_attn_prefix = False,
#     rel_pos_bias = True,
#     flash_attn = False,
#     **kwargs
#     '''

#     m = Transformer(dim=64, depth=4, heads=4)
#     '''
#     self,
#     x,
#     self_attn_mask = None,
#     context = None,
#     context_mask = None,
#     attn_bias = None,
#     return_kv_cache = False,
#     kv_cache = None
#     '''
#     input = torch.rand(8, 200, 64) # b, h, i, d
#     output = m(x=input)
#     print(output.shape)
