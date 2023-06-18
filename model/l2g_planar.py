import numpy as np
import os,sys,time
import torch
import torch.nn.functional as torch_F
import torchvision
import torchvision.transforms.functional as torchvision_F
import tqdm
from easydict import EasyDict as edict
import PIL
import PIL.Image,PIL.ImageDraw
import imageio

import util,util_vis
from util import log,debug
from . import base
import warp
import roma
from kornia.geometry.homography import find_homography_dlt
# ============================ main engine for training and evaluation ============================

class Model(base.Model):

    def __init__(self,opt):
        super().__init__(opt)
        opt.H_crop,opt.W_crop = opt.data.patch_crop

    def load_dataset(self,opt,eval_split=None):
        image_raw = PIL.Image.open(opt.data.image_fname).convert('RGB')
        self.image_raw = torchvision_F.to_tensor(image_raw).to(opt.device)

    def build_networks(self,opt):
        super().build_networks(opt)
        self.graph.warp_embedding = torch.nn.Embedding(opt.batch_size,opt.arch.embedding_dim).to(opt.device)
        self.graph.warp_mlp = localWarp(opt).to(opt.device)

    def setup_optimizer(self,opt):
        log.info("setting up optimizers...")
        optim_list = [
            dict(params=self.graph.neural_image.parameters(),lr=opt.optim.lr),
        ]
        optimizer = getattr(torch.optim,opt.optim.algo)
        self.optim = optimizer(optim_list)
        # set up scheduler
        if opt.optim.sched:
            scheduler = getattr(torch.optim.lr_scheduler,opt.optim.sched.type)
            if opt.optim.lr_end:
                assert(opt.optim.sched.type=="ExponentialLR")
                opt.optim.sched.gamma = (opt.optim.lr_end/opt.optim.lr)**(1./opt.max_iter)
            kwargs = { k:v for k,v in opt.optim.sched.items() if k!="type" }
            self.sched = scheduler(self.optim,**kwargs)

        # warp 
        self.optim_warp = optimizer([dict(params=self.graph.warp_embedding.parameters(),lr=opt.optim.lr_warp)])
        self.optim_warp.add_param_group(dict(params=self.graph.warp_mlp.parameters(),lr=opt.optim.lr_warp))
        # set up scheduler
        if opt.optim.sched_warp:
            scheduler = getattr(torch.optim.lr_scheduler,opt.optim.sched_warp.type)
            if opt.optim.lr_warp_end:
                assert(opt.optim.sched_warp.type=="ExponentialLR")
                opt.optim.sched_warp.gamma = (opt.optim.lr_warp_end/opt.optim.lr_warp)**(1./opt.max_iter)
            kwargs = { k:v for k,v in opt.optim.sched_warp.items() if k!="type" }
            self.sched_warp = scheduler(self.optim_warp,**kwargs)


    def setup_visualizer(self,opt):
        super().setup_visualizer(opt)
        # set colors for visualization
        box_colors = ["#ff0000","#40afff","#9314ff","#ffd700","#00ff00"]
        box_colors = list(map(util.colorcode_to_number,box_colors))
        self.box_colors = np.array(box_colors).astype(int)
        assert(len(self.box_colors)==opt.batch_size)
        # create visualization directory
        self.vis_path = "{}/vis".format(opt.output_path)
        os.makedirs(self.vis_path,exist_ok=True)
        self.video_fname = "{}/vis.mp4".format(opt.output_path)

    def train(self,opt):
        # before training
        log.title("TRAINING START")
        self.timer = edict(start=time.time(),it_mean=None)
        self.ep = self.it = self.vis_it = 0
        self.graph.train()
        var = edict(idx=torch.arange(opt.batch_size))
        # pre-generate perturbations
        self.homo_pert, self.rot_pert, self.trans_pert, var.image_pert = self.generate_warp_perturbation(opt)
        # train
        var = util.move_to_device(var,opt.device)
        loader = tqdm.trange(opt.max_iter,desc="training",leave=False)
        # visualize initial state
        var = self.graph.forward(opt,var)
        self.visualize(opt,var,step=0)
        for it in loader:
            # train iteration
            loss = self.train_iteration(opt,var,loader)
        # after training
        os.system("ffmpeg -y -framerate 30 -i {}/%d.png -pix_fmt yuv420p {}".format(self.vis_path,self.video_fname))
        self.save_checkpoint(opt,ep=None,it=self.it)
        if opt.tb:
            self.tb.flush()
            self.tb.close()
        if opt.visdom: self.vis.close()
        log.title("TRAINING DONE")

    def train_iteration(self,opt,var,loader):
        self.optim_warp.zero_grad()
        loss = super().train_iteration(opt,var,loader)
        self.graph.neural_image.progress.data.fill_(self.it/opt.max_iter)
        self.optim_warp.step()
        if opt.optim.sched_warp: self.sched_warp.step()
        if opt.optim.sched: self.sched.step()
        return loss

    def generate_warp_perturbation(self,opt):
        # pre-generate perturbations (translational noise + homography noise)
        def create_random_perturbation(batch_size):
            if opt.warp.dof==1:
                homo_pert = torch.zeros(batch_size,8,device=opt.device)*opt.warp.noise_h
                rot_pert = torch.randn(batch_size,1,device=opt.device)*opt.warp.noise_r
                trans_pert = torch.zeros(batch_size,2,device=opt.device)*opt.warp.noise_t
            elif opt.warp.dof==2:
                homo_pert = torch.zeros(batch_size,8,device=opt.device)*opt.warp.noise_h
                rot_pert = torch.zeros(batch_size,1,device=opt.device)*opt.warp.noise_r
                trans_pert = torch.randn(batch_size,2,device=opt.device)*opt.warp.noise_t
            elif opt.warp.dof==3:
                homo_pert = torch.zeros(batch_size,8,device=opt.device)*opt.warp.noise_h
                rot_pert = torch.randn(batch_size,1,device=opt.device)*opt.warp.noise_r
                trans_pert = torch.randn(batch_size,2,device=opt.device)*opt.warp.noise_t
            elif opt.warp.dof==8:
                homo_pert = torch.randn(batch_size,8,device=opt.device)*opt.warp.noise_h
                homo_pert[:,:2]=0
                rot_pert = torch.randn(batch_size,1,device=opt.device)*opt.warp.noise_r
                trans_pert = torch.randn(batch_size,2,device=opt.device)*opt.warp.noise_t
            else: assert(False)
            return homo_pert, rot_pert, trans_pert

        homo_pert = torch.zeros(opt.batch_size,8,device=opt.device)
        rot_pert = torch.zeros(opt.batch_size,1,device=opt.device)
        trans_pert = torch.zeros(opt.batch_size,2,device=opt.device)
        for i in range(opt.batch_size):
            homo_pert_i, rot_pert_i, trans_pert_i = create_random_perturbation(1)
            while not warp.check_corners_in_range_compose(opt, homo_pert_i, rot_pert_i, trans_pert_i):
                homo_pert_i, rot_pert_i, trans_pert_i = create_random_perturbation(1)
            homo_pert[i], rot_pert[i], trans_pert[i] = homo_pert_i, rot_pert_i, trans_pert_i

        if opt.warp.fix_first:
            homo_pert[0],rot_pert[0],trans_pert[0] = 0,0,0

        # create warped image patches
        xy_grid = warp.get_normalized_pixel_grid_crop(opt) # [B,HW,2]

        xy_grid_hom = warp.camera.to_hom(xy_grid)
        warp_matrix = warp.lie.sl3_to_SL3(homo_pert)
        warped_grid_hom = xy_grid_hom@warp_matrix.transpose(-2,-1)
        xy_grid_warped = warped_grid_hom[...,:2]/(warped_grid_hom[...,2:]+1e-8) # [B,HW,2]
        warp_matrix = warp.lie.so2_to_SO2(rot_pert)
        xy_grid_warped = xy_grid_warped@warp_matrix.transpose(-2,-1) # [B,HW,2]
        xy_grid_warped = xy_grid_warped+trans_pert[...,None,:]

        xy_grid_warped = xy_grid_warped.view([opt.batch_size,opt.H_crop,opt.W_crop,2])
        xy_grid_warped = torch.stack([xy_grid_warped[...,0]*max(opt.H,opt.W)/opt.W,
                                      xy_grid_warped[...,1]*max(opt.H,opt.W)/opt.H],dim=-1)
        image_raw_batch = self.image_raw.repeat(opt.batch_size,1,1,1)
        image_pert_all = torch_F.grid_sample(image_raw_batch,xy_grid_warped,align_corners=False)

        return homo_pert, rot_pert, trans_pert, image_pert_all

    def visualize_patches_compose(self,opt, homo_pert, rot_pert, trans_pert):
        image_pil = torchvision_F.to_pil_image(self.image_raw).convert("RGBA")
        draw_pil = PIL.Image.new("RGBA",image_pil.size,(0,0,0,0))
        draw = PIL.ImageDraw.Draw(draw_pil)
        corners_all = warp.warp_corners_compose(opt, homo_pert, rot_pert, trans_pert)
        corners_all[...,0] = (corners_all[...,0]/opt.W*max(opt.H,opt.W)+1)/2*opt.W-0.5
        corners_all[...,1] = (corners_all[...,1]/opt.H*max(opt.H,opt.W)+1)/2*opt.H-0.5
        for i,corners in enumerate(corners_all):
            P = [tuple(float(n) for n in corners[j]) for j in range(4)]
            draw.line([P[0],P[1],P[2],P[3],P[0]],fill=tuple(self.box_colors[i]),width=3)
        image_pil.alpha_composite(draw_pil)
        image_tensor = torchvision_F.to_tensor(image_pil.convert("RGB"))
        return image_tensor

    def visualize_patches_use_matrix(self,opt,warp_matrix):
        image_pil = torchvision_F.to_pil_image(self.image_raw).convert("RGBA")
        draw_pil = PIL.Image.new("RGBA",image_pil.size,(0,0,0,0))
        draw = PIL.ImageDraw.Draw(draw_pil)
        corners_all = warp.warp_corners_use_matrix(opt,warp_matrix)
        corners_all[...,0] = (corners_all[...,0]/opt.W*max(opt.H,opt.W)+1)/2*opt.W-0.5
        corners_all[...,1] = (corners_all[...,1]/opt.H*max(opt.H,opt.W)+1)/2*opt.H-0.5
        for i,corners in enumerate(corners_all):
            P = [tuple(float(n) for n in corners[j]) for j in range(4)]
            draw.line([P[0],P[1],P[2],P[3],P[0]],fill=tuple(self.box_colors[i]),width=3)
        image_pil.alpha_composite(draw_pil)
        image_tensor = torchvision_F.to_tensor(image_pil.convert("RGB"))
        return image_tensor

    @torch.no_grad()
    def predict_entire_image(self,opt):
        xy_grid = warp.get_normalized_pixel_grid(opt)[:1]
        rgb = self.graph.neural_image.forward(opt,xy_grid) # [B,HW,3]
        image = rgb.view(opt.H,opt.W,3).detach().cpu().permute(2,0,1)
        return image

    @torch.no_grad()
    def log_scalars(self,opt,var,loss,metric=None,step=0,split="train"):
        super().log_scalars(opt,var,loss,metric=metric,step=step,split=split)
        # compute PSNR
        psnr = -10*loss.render.log10()
        self.tb.add_scalar("{0}/{1}".format(split,"PSNR"),psnr,step)
        # warp error
        pred_corners = warp.warp_corners_use_matrix(opt,self.graph.warp_matrix)
        pred_corners[...,0] = (pred_corners[...,0]/opt.W*max(opt.H,opt.W)+1)/2*opt.W-0.5
        pred_corners[...,1] = (pred_corners[...,1]/opt.H*max(opt.H,opt.W)+1)/2*opt.H-0.5
        
        gt_corners = warp.warp_corners_compose(opt, self.homo_pert, self.rot_pert, self.trans_pert)
        gt_corners[...,0] = (gt_corners[...,0]/opt.W*max(opt.H,opt.W)+1)/2*opt.W-0.5
        gt_corners[...,1] = (gt_corners[...,1]/opt.H*max(opt.H,opt.W)+1)/2*opt.H-0.5

        warp_error = (pred_corners-gt_corners).norm(dim=-1).mean()
        self.tb.add_scalar("{0}/{1}".format(split,"warp error"),warp_error,step)


    @torch.no_grad()
    def visualize(self,opt,var,step=0,split="train"):
        # dump frames for writing to video
        frame_GT = self.visualize_patches_compose(opt,self.homo_pert, self.rot_pert, self.trans_pert)
        frame = self.visualize_patches_use_matrix(opt,self.graph.warp_matrix)
        frame2 = self.predict_entire_image(opt)
        frame_cat = (torch.cat([frame,frame2],dim=1)*255).byte().permute(1,2,0).numpy()
        imageio.imsave("{}/{}.png".format(self.vis_path,self.vis_it),frame_cat)
        self.vis_it += 1
        # visualize in Tensorboard
        if opt.tb:
            colors = self.box_colors
            util_vis.tb_image(opt,self.tb,step,split,"image_pert",util_vis.color_border(var.image_pert,colors))
            util_vis.tb_image(opt,self.tb,step,split,"rgb_warped",util_vis.color_border(var.rgb_warped_map,colors))
            util_vis.tb_image(opt,self.tb,self.it+1,"train","image_boxes",frame[None])
            util_vis.tb_image(opt,self.tb,self.it+1,"train","image_boxes_GT",frame_GT[None])
            util_vis.tb_image(opt,self.tb,self.it+1,"train","image_entire",frame2[None])
            local_warp_field = torch.cat([(var.local_warp_field+1)/2,torch.zeros_like(var.local_warp_field[:,:1,...])],dim=1) # [B,3,H,W]
            global_warp_field = torch.cat([(var.global_warp_field+1)/2,torch.zeros_like(var.global_warp_field[:,:1,...])],dim=1) # [B,3,H,W]            
            util_vis.tb_image(opt,self.tb,step,split,"warp_field_local",util_vis.color_border(local_warp_field,colors))
            util_vis.tb_image(opt,self.tb,step,split,"warp_field_global",util_vis.color_border(global_warp_field,colors))

# ============================ computation graph for forward/backprop ============================

class Graph(base.Graph):

    def __init__(self,opt):
        super().__init__(opt)
        self.neural_image = NeuralImageFunction(opt)

    def forward(self,opt,var,mode=None):
        # warp
        xy_grid = warp.get_normalized_pixel_grid_crop(opt)
        warp_embedding = self.warp_embedding.weight[:,None,:].expand(-1,xy_grid.shape[1],-1)
        local_warp_param = self.warp_mlp(opt,torch.cat((xy_grid,warp_embedding),dim=-1))
        if opt.warp.fix_first:
            local_warp_param[0] = 0

        if opt.warp.dof==1:
            local_warp_matrix = warp.lie.so2_to_SO2(local_warp_param)
            var.local_warped_grid = (xy_grid[...,None,:]@local_warp_matrix.transpose(-2,-1))[...,0,:] # [B,HW,2]
            self.warp_matrix = roma.rigid_vectors_registration(xy_grid, var.local_warped_grid)
            var.global_warped_grid = xy_grid@self.warp_matrix.transpose(-2,-1) # [B,HW,2]
        elif opt.warp.dof==2:
            var.local_warped_grid = xy_grid + local_warp_param
            self.warp_matrix = var.local_warped_grid.mean(-2)-xy_grid.mean(-2)
            var.global_warped_grid = xy_grid + self.warp_matrix[...,None,:]
        elif opt.warp.dof==3:
            xy_grid_hom = warp.camera.to_hom(xy_grid[...,None,:])
            local_warp_matrix = warp.lie.se2_to_SE2(local_warp_param)
            var.local_warped_grid = (xy_grid_hom@local_warp_matrix.transpose(-2,-1))[...,0,:] # [B,HW,2]
            R_global, t_global = roma.rigid_points_registration(xy_grid, var.local_warped_grid)
            self.warp_matrix = torch.cat((R_global,t_global[...,None]),-1)
            xy_grid_hom = warp.camera.to_hom(xy_grid)
            var.global_warped_grid = xy_grid_hom@self.warp_matrix.transpose(-2,-1) # [B,HW,2]
        elif opt.warp.dof==8:
            xy_grid_hom = warp.camera.to_hom(xy_grid[...,None,:])
            local_warp_matrix = warp.lie.se2_to_SE2(local_warp_param)
            var.local_warped_grid = (xy_grid_hom@local_warp_matrix.transpose(-2,-1))[...,0,:] # [B,HW,2]
            self.warp_matrix = find_homography_dlt(xy_grid, var.local_warped_grid)
            xy_grid_hom = warp.camera.to_hom(xy_grid)
            global_warped_grid = xy_grid_hom@self.warp_matrix.transpose(-2,-1)
            var.global_warped_grid = global_warped_grid[...,:2]/(global_warped_grid[...,2:]+1e-8) # [B,HW,2]

        # render images
        var.rgb_warped = self.neural_image.forward(opt,var.local_warped_grid) # [B,HW,3]
        var.rgb_warped_map = var.rgb_warped.view(opt.batch_size,opt.H_crop,opt.W_crop,3).permute(0,3,1,2) # [B,3,H,W]
        var.local_warp_field = var.local_warped_grid.view(opt.batch_size,opt.H_crop,opt.W_crop,2).permute(0,3,1,2) # [B,2,H,W]
        var.global_warp_field = var.global_warped_grid.view(opt.batch_size,opt.H_crop,opt.W_crop,2).permute(0,3,1,2) # [B,2,H,W]
        return var

    def compute_loss(self,opt,var,mode=None):
        loss = edict()
        if opt.loss_weight.render is not None:
            image_pert = var.image_pert.view(opt.batch_size,3,opt.H_crop*opt.W_crop).permute(0,2,1)
            loss.render = self.MSE_loss(var.rgb_warped,image_pert)
        if opt.loss_weight.global_alignment is not None:
            loss.global_alignment = self.MSE_loss(var.local_warped_grid,var.global_warped_grid)
        return loss

class NeuralImageFunction(torch.nn.Module):

    def __init__(self,opt):
        super().__init__()
        self.define_network(opt)
        self.progress = torch.nn.Parameter(torch.tensor(0.)) # use Parameter so it could be checkpointed

    def define_network(self,opt):
        input_2D_dim = 2+4*opt.arch.posenc.L_2D if opt.arch.posenc else 2
        # point-wise RGB prediction
        self.mlp = torch.nn.ModuleList()
        L = util.get_layer_dims(opt.arch.layers)
        for li,(k_in,k_out) in enumerate(L):
            if li==0: k_in = input_2D_dim
            if li in opt.arch.skip: k_in += input_2D_dim
            linear = torch.nn.Linear(k_in,k_out)
            if opt.barf_c2f and li==0:
                # rescale first layer init (distribution was for pos.enc. but only xy is first used)
                scale = np.sqrt(input_2D_dim/2.)
                linear.weight.data *= scale
                linear.bias.data *= scale
            self.mlp.append(linear)

    def forward(self,opt,coord_2D): # [B,...,3]
        if opt.arch.posenc:
            points_enc = self.positional_encoding(opt,coord_2D,L=opt.arch.posenc.L_2D)
            points_enc = torch.cat([coord_2D,points_enc],dim=-1) # [B,...,6L+3]
        else: points_enc = coord_2D
        feat = points_enc
        # extract implicit features
        for li,layer in enumerate(self.mlp):
            if li in opt.arch.skip: feat = torch.cat([feat,points_enc],dim=-1)
            feat = layer(feat)
            if li!=len(self.mlp)-1:
                feat = torch_F.relu(feat)
        rgb = feat.sigmoid_() # [B,...,3]
        return rgb

    def positional_encoding(self,opt,input,L): # [B,...,N]
        shape = input.shape
        freq = 2**torch.arange(L,dtype=torch.float32,device=opt.device)*np.pi # [L]
        spectrum = input[...,None]*freq # [B,...,N,L]
        sin,cos = spectrum.sin(),spectrum.cos() # [B,...,N,L]
        input_enc = torch.stack([sin,cos],dim=-2) # [B,...,N,2,L]
        input_enc = input_enc.view(*shape[:-1],-1) # [B,...,2NL]
        # coarse-to-fine: smoothly mask positional encoding for BARF
        if opt.barf_c2f is not None:
            # set weights for different frequency bands
            start,end = opt.barf_c2f
            alpha = (self.progress.data-start)/(end-start)*L
            k = torch.arange(L,dtype=torch.float32,device=opt.device)
            weight = (1-(alpha-k).clamp_(min=0,max=1).mul_(np.pi).cos_())/2
            # apply weights
            shape = input_enc.shape
            input_enc = (input_enc.view(-1,L)*weight).view(*shape)
        return input_enc

class localWarp(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()
        # point-wise se3 prediction
        input_2D_dim = 2
        self.mlp_warp = torch.nn.ModuleList()
        L = util.get_layer_dims(opt.arch.layers_warp)
        for li,(k_in,k_out) in enumerate(L):
            if li==0: k_in = input_2D_dim+opt.arch.embedding_dim
            if li in opt.arch.skip_warp: k_in += input_2D_dim+opt.arch.embedding_dim
            linear = torch.nn.Linear(k_in,k_out)
            self.mlp_warp.append(linear)

    def forward(self,opt,uvf):
        feat = uvf
        for li,layer in enumerate(self.mlp_warp):
            if li in opt.arch.skip_warp: feat = torch.cat([feat,uvf],dim=-1)
            feat = layer(feat)
            if li!=len(self.mlp_warp)-1:
                feat = torch_F.relu(feat)
        warp = feat
        return warp