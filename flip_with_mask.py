import torch

def flip_with_mask(x, lengths, mask=None): # flips tensor but keeps masked values at the end of the sequence
    # x: (batch, seq_len, d)
    # lengths: (batch,)
    # mask: (batch, seq_len)
    max_len = lengths.max()
    mask = ~(torch.arange(max_len, device=lengths.device) < lengths.unsqueeze(1)) if mask is None else mask
    selection_indices = torch.arange(x.size(1) - 1, -1, -1)[None].expand(x.size()[:-1])
    max_len = lengths.max()
    offset = (max_len - lengths)[:, None].expand(x.size()[:-1]) # offset based on difference from max length
    selection_indices = selection_indices - offset # take offset from indices
    selection_indices[mask] = max_len - 1 # set values that are out of range to the last index
    x = x.gather(1, selection_indices.unsqueeze(-1).expand(x.size()))
    return x


if __name__ == '__main__':
    # example
    lengths = torch.LongTensor([5, 8, 9, 12])
    x = torch.arange(12)[None].repeat(4, 1) + 1
    mask = ~(torch.arange(x.size(1)).expand(x.size()) < lengths.unsqueeze(1))
    x.masked_fill_(mask, 0)
    # add dim
    d = 256
    x = x[..., None].repeat(1, 1, d)
    print(x.shape)
    f = flip_with_mask(x, lengths)
    print('before:')
    print(x[..., 0])
    print('after:')
    print(f[..., 0])
