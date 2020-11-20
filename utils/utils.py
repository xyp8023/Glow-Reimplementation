from math import log
# from torch import log 
import torch

# def generative_loss(logp, logdet, img_size):
#     a = 2 ** 8
#     num_pixels = img_size * img_size * 3
    
#     loss = -log(a) * num_pixels + logp + logdet

#     # loss = (-loss / (log(2) * num_pixels)).sum()
#     # logp = (logp / (log(2) * num_pixels)).sum()
#     # logdet = (logdet / (log(2) * num_pixels)).sum()
    
#     loss = (-loss / (log(2) * num_pixels)).mean()
#     logp = (logp / (log(2) * num_pixels)).mean()
#     logdet = (logdet / (log(2) * num_pixels)).mean()
    
#     return loss, logp, logdet

def generative_loss(logp, logdet, img_size, n_bits=8):
    a = 2 ** n_bits
    num_pixels = img_size * img_size * 3
    
    # loss = -log(a) * num_pixels + logp.mean() + logdet.mean()
    loss = -log(a) * num_pixels + logp.mean() + logdet.mean()

    # loss = (-loss / (log(2) * num_pixels)).sum()
    # logp = (logp / (log(2) * num_pixels)).sum()
    # logdet = (logdet / (log(2) * num_pixels)).sum()
    
    loss = (-loss / (log(2) * num_pixels))
    logp = (logp.mean() / (log(2) * num_pixels))
    logdet = (logdet.mean() / (log(2) * num_pixels))
    # print(logdet)
    return loss, logp, logdet

def random_sample(N, C=3, H=32, W=32, n_blocks=3, n_flows=32, temp=0.8):
    z_outs_random_sample = []
    for i_b in range(n_blocks-1):
        z_outs_random_sample.append(temp*torch.randn(N, C*2**(i_b+1), H//2**(i_b+1), W//2**(i_b+1)).cuda())

    z_outs_random_sample.append(temp*torch.randn(N,C*2**(n_blocks+1),H//2**(n_blocks),W//2**(n_blocks)).cuda())
    return z_outs_random_sample

def linearInterpolate(glowmodel, img1, img2, interpo_num=39):
    _, _, z1 = glowmodel(img1)
    _, _, z2 = glowmodel(img2)
    zspace = []
    for i in range(len(z1)):
        single_interpo = torch.zeros(interpo_num+1, z1[i].shape[1], z1[i].shape[2], z1[i].shape[3]).to(device)
        single_interpo[0, :] = z1[i]
        single_interpo[-1, :] = z2[i]
        print(single_interpo.shape)
        for interpo in range(interpo_num):
            c = torch.lerp(z1[i], z2[i], interpo/interpo_num)
            single_interpo[interpo,:] = c
        zspace.append(single_interpo)
    return glowmodel.reverse(zspace).cpu().data

if __name__ == "__main__":
    pass