import torch

def flip_with_mask(x, lengths, mask=None): # flips tensor but keeps masked values at the end of the sequence
    mask = ~(torch.arange(x.size(1)).expand(x.size()) < lengths.unsqueeze(1)) if mask is None else mask
    selection_indices = torch.arange(x.size(1) - 1, -1, -1)[None].expand(x.size())
    max_len = lengths.max()
    offset = (max_len - lengths)[:, None].expand(x.size()) # offset based on difference from max length
    selection_indices = selection_indices - offset # take offset from indices
    selection_indices[mask] = max_len - 1 # set values that are out of range to the last index
    x = x.gather(1, selection_indices)
    return x


if __name__ == '__main__':
    # example
    lengths = torch.LongTensor([5, 8, 9, 12])
    x = torch.arange(12)[None].repeat(4, 1) + 1
    mask = ~(torch.arange(x.size(1)).expand(x.size()) < lengths.unsqueeze(1))
    x.masked_fill_(mask, 0)
    f = flip_with_mask(x, lengths)
    print('before:')
    print(x)
    print('after:')
    print(f)
