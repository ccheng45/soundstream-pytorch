import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from utils import CfgNode as CN
from transformer_einsops import TransformerBlockEnins
class GPT(nn.Module):
    """baby gpt like model with transformer blocks """

    @staticmethod
    def get_default_config():
        C = CN()
        
        C.n_layer = 12
        C.n_head = 16
        C.n_embd =  1024
        # these options must be filled in externally
        C.vocab_size = 320
        # C.block_size = None
        # dropout hyperparameters
        C.embd_pdrop = 0.1
        C.resid_pdrop = 0.1
        C.attn_pdrop = 0.1
        return C

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        # assert config.block_size is not None
        # self.block_size = config.block_size

        params_given = all([config.n_layer is not None, config.n_head is not None, config.n_embd is not None])
        assert params_given # exactly one of these (XOR)
        

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            # wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.embd_pdrop),
            h = nn.ModuleList([TransformerBlockEnins(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # init all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("number of parameters: %.2fM" % (n_params/1e6,))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    # @classmethod
    # def from_pretrained(cls, model_type):
    #     """
    #     Initialize a pretrained GPT model by copying over the weights
    #     from a huggingface/transformers checkpoint.
    #     """
    #     assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
    #     from transformers import GPT2LMHeadModel

    #     # create a from-scratch initialized minGPT model
    #     config = cls.get_default_config()
    #     config.model_type = model_type
    #     config.vocab_size = 50257 # openai's model vocabulary
    #     config.block_size = 1024  # openai's model block_size
    #     model = GPT(config)
    #     sd = model.state_dict()

    #     # init a huggingface/transformers model
    #     model_hf = GPT2LMHeadModel.from_pretrained(model_type)
    #     sd_hf = model_hf.state_dict()

    #     # copy while ensuring all of the parameters are aligned and match in names and shapes
    #     keys = [k for k in sd_hf if not k.endswith('attn.masked_bias')] # ignore these
    #     transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
    #     # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla nn.Linear.
    #     # this means that we have to transpose these weights when we import them
    #     assert len(keys) == len(sd)
    #     for k in keys:
    #         if any(k.endswith(w) for w in transposed):
    #             # special treatment for the Conv1D weights we need to transpose
    #             assert sd_hf[k].shape[::-1] == sd[k].shape
    #             with torch.no_grad():
    #                 sd[k].copy_(sd_hf[k].t())
    #         else:
    #             # vanilla copy over the other parameters
    #             assert sd_hf[k].shape == sd[k].shape
    #             with torch.no_grad():
    #                 sd[k].copy_(sd_hf[k])

    #     return model

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        # decay = set()
        # no_decay = set()
        # whitelist_weight_modules = (torch.nn.Linear, )
        # blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        # for mn, m in self.named_modules():
        #     for pn, p in m.named_parameters():
        #         fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
        #         # random note: because named_modules and named_parameters are recursive
        #         # we will see the same tensors p many many times. but doing it this way
        #         # allows us to know which parent module any tensor p belongs to...
        #         if pn.endswith('bias'):
        #             # all biases will not be decayed
        #             no_decay.add(fpn)
        #         elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
        #             # weights of whitelist modules will be weight decayed
        #             decay.add(fpn)
        #         elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
        #             # weights of blacklist modules will NOT be weight decayed
        #             no_decay.add(fpn)

        # # validate that we considered every parameter
        # param_dict = {pn: p for pn, p in self.named_parameters()}
        # inter_params = decay & no_decay
        # union_params = decay | no_decay
        # assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        # assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
        #                                             % (str(param_dict.keys() - union_params), )

        # # create the pytorch optimizer object
        # optim_groups = [
        #     {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
        #     {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        # ]
        # optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        optimizer = torch.optim.AdamW(lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self, idx, targets=None):
        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        x = self.transformer.drop(tok_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss

    # @torch.no_grad()
    # def generate(self, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
    #     """
    #     Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
    #     the sequence max_new_tokens times, feeding the predictions back into the model each time.
    #     Most likely you'll want to make sure to be in model.eval() mode of operation for this.
    #     """
    #     for _ in range(max_new_tokens):
    #         # if the sequence context is growing too long we must crop it at block_size
    #         idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
    #         # forward the model to get the logits for the index in the sequence
    #         logits, _ = self(idx_cond)
    #         # pluck the logits at the final step and scale by desired temperature
    #         logits = logits[:, -1, :] / temperature
    #         # optionally crop the logits to only the top k options
    #         if top_k is not None:
    #             v, _ = torch.topk(logits, top_k)
    #             logits[logits < v[:, [-1]]] = -float('Inf')
    #         # apply softmax to convert logits to (normalized) probabilities
    #         probs = F.softmax(logits, dim=-1)
    #         # either sample from the distribution or take the most likely element
    #         if do_sample:
    #             idx_next = torch.multinomial(probs, num_samples=1)
    #         else:
    #             _, idx_next = torch.topk(probs, k=1, dim=-1)
    #         # append sampled index to the running sequence and continue
    #         idx = torch.cat((idx, idx_next), dim=1)

    #     return idx