###########################
# Latent ODEs for Irregularly-Sampled Time Series
# Author: Yulia Rubanova
###########################

import os
import numpy as np

import torch
import torch.nn as nn
from torch.nn.functional import relu

import rnn_utils as utils
from latent_ode_dit import LatentODE
from encoder_decoder import *
from diffeq_solver import DiffeqSolver

from torch.distributions.normal import Normal
from ode_func import ODEFunc, ODEFunc_w_Poisson
from run_dnerf_helpers import LatentNetwork, get_embedder
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp

# from anode_models import ODEBlock, ODEFunc2

# DiT Block with adaptive layer norm zero (adaLN-Zero) conditioning.
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    # return x * (1 + scale) + shift
class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

#####################################################################################################

def create_LatentODE_model(input_dim, z0_prior, obsrv_std, device, 
    classif_per_tp = False, n_labels = 1, latents=64, units=100, gen_layers=1,
    rec_layers=1, rec_dims=20, z0_encoder="mlp", gru_units=100, poisson=False, 
    num_frames=80, time_invariant=True):

    embed_angle, angle_dim = get_embedder(10, 2, 0)
    embed_vel, vel_dim = get_embedder(10, 2, 0)
    latent_embedder_out_dim = angle_dim

    print("Latent Embedder Size: ",latent_embedder_out_dim)
    input_dim = latent_embedder_out_dim
    dim = latent_embedder_out_dim

    if poisson:
        lambda_net = utils.create_net(dim, input_dim, 
            n_layers = 1, n_units = units, nonlinear = nn.Tanh)

        # ODE function produces the gradient for latent state and for poisson rate
        ode_func_net = utils.create_net(dim * 2, latents * 2, 
            n_layers = gen_layers, n_units = units, nonlinear = nn.Tanh)

        gen_ode_func = ODEFunc_w_Poisson(
            input_dim = input_dim, 
            latent_dim = latents * 2,
            ode_func_net = ode_func_net,
            lambda_net = lambda_net,
            device = device).to(device)
    else:
        latents_new = 64
        linear = None
        #linear = nn.Linear(latents_new, 20)  ## ADDED
        #latents_new = 20  ## ADDED
        dim = latents_new
        
        ode_func_net = utils.create_net(dim, latents_new, 
            n_layers = gen_layers, n_units = units, nonlinear = nn.Tanh)


        gen_ode_func = ODEFunc(
            input_dim = input_dim, 
            latent_dim = latents_new, 
            ode_func_net = ode_func_net,
            device = device).to(device)


    angle_dit = DiTBlock(hidden_size=64, num_heads=4)
    vel_dit = DiTBlock(hidden_size=64, num_heads=4)

    z0_diffeq_solver = None
    n_rec_dims = rec_dims
    #enc_input_dim = int(input_dim) * 2 # we concatenate the mask
    enc_input_dim = int(input_dim)  #### OG Code
    #enc_input_dim = int(input_dim) + 512
    gen_data_dim = input_dim

    z0_dim = latents
    if poisson:
        z0_dim += latents # predict the initial poisson rate

    if z0_encoder == "odernn":
        ode_func_net = utils.create_net(n_rec_dims, n_rec_dims, 
             n_layers = rec_layers, n_units = units, nonlinear = nn.Tanh)


        rec_ode_func = ODEFunc(
            input_dim = enc_input_dim,
            latent_dim = n_rec_dims,
            ode_func_net = ode_func_net,
            device = device).to(device)

        z0_diffeq_solver = DiffeqSolver(enc_input_dim, rec_ode_func, "euler", latents, 
            odeint_rtol = 1e-3, odeint_atol = 1e-4, device = device)
        
        encoder_z0 = Encoder_z0_ODE_RNN(n_rec_dims, enc_input_dim, z0_diffeq_solver, 
            z0_dim = z0_dim, n_gru_units = gru_units, device = device).to(device)
        
    
    if z0_encoder == "mlp":
        W = 256
        layers = [nn.Linear(enc_input_dim, W)]
        for i in range(7):
            
            layer = nn.Linear

            in_channels = W
            if i == 4:
                in_channels += enc_input_dim

            layers += [layer(in_channels, W)]
        
        layers += [layer(W, latents)]

        # feature = nn.Linear()
        

        encoder_z0 = nn.ModuleList(layers)


        W = 256
        layers = [nn.Linear(vel_dim, W)]
        for i in range(7):
            
            layer = nn.Linear

            in_channels = W
            if i == 4:
                in_channels += vel_dim

            layers += [layer(in_channels, W)]
        
        layers += [layer(W, latents)]        

        encoder_z0_vel = nn.ModuleList(layers)


    elif z0_encoder == "linear":
        encoder_z0 = nn.Linear(enc_input_dim, latents)
        


    
    if z0_encoder == 'vae':
        decoder = VAEDecoder(latent_dim=latents_new, hidden_dim=256, output_dim=512)
    else:
        # decoder = Decoder(latents_new, 512).to(device)
        decoder = Decoder(64, 512).to(device)
        # decoder = Decoder(512+augment_dim, 512).to(device) ## Anode
    # decoder_pose = Decoder(latents_new, angle_dim).to(device)
    decoder_pose = Decoder(64, angle_dim).to(device)
    # decoder_vel = Decoder(vel_dim, vel_dim).to(device)
    decoder_vel = Decoder(64, vel_dim).to(device)


    diffeq_solver = DiffeqSolver(0, gen_ode_func, 'euler', latents_new, 
        odeint_rtol = 1e-3, odeint_atol = 1e-4, device = device)
 
    
    model = LatentODE(
        input_dim = gen_data_dim, 
        latent_dim = latents, 
        encoder_z0 = encoder_z0,
        decoder = decoder, 
        diffeq_solver = diffeq_solver,
        z0_prior = z0_prior, 
        device = device,
        obsrv_std = obsrv_std,
        use_poisson_proc = poisson, 
        use_binary_classif = False,
        linear_classifier = False,
        classif_per_tp = classif_per_tp,
        n_labels = n_labels,
        train_classif_w_reconstr = False,
        num_frames=num_frames,
        latent_embedder_out_dim = latent_embedder_out_dim,
        latent_embedder = None,
        embed_angle = embed_angle,
        embed_vel = embed_vel,
        decoder_pose = decoder_pose,
        decoder_vel = decoder_vel,
        z0_encoder_type = z0_encoder,
        linear = linear, 
        encoder_z0_vel = encoder_z0_vel,
        angle_dit=angle_dit,
        vel_dit = vel_dit
        ).to(device)

    return model