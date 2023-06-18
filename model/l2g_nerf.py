import numpy as np
import os,sys,time
import torch
import torch.nn.functional as torch_F
import torchvision
import torchvision.transforms.functional as torchvision_F
import tqdm
from easydict import EasyDict as edict
import visdom
import matplotlib.pyplot as plt

import util,util_vis
from util import log,debug
from . import nerf
import camera

import roma
# ============================ main engine for training and evaluation ============================

class Model(nerf.Model):

    def __init__(self,opt):
        super().__init__(opt)

    def build_networks(self,opt):
        super().build_networks(opt)
        if opt.camera.noise:
            # pre-generate synthetic pose perturbation
            so3_noise = torch.randn(len(self.train_data),3,device=opt.device)*opt.camera.noise_r
            t_noise = torch.randn(len(self.train_data),3,device=opt.device)*opt.camera.noise_t
            self.graph.pose_noise = torch.cat([camera.lie.so3_to_SO3(so3_noise),t_noise[...,None]],dim=-1) # [...,3,4]

        self.graph.warp_embedding = torch.nn.Embedding(len(self.train_data),opt.arch.embedding_dim).to(opt.device)
        self.graph.warp_mlp = localWarp(opt).to(opt.device)

        pose_GT = self.train_data.get_all_camera_poses(opt).to(opt.device)
        # add synthetic pose perturbation to all training data
        if opt.data.dataset=="blender":
            pose = pose_GT
            if opt.camera.noise:
                pose = camera.pose.compose([pose, self.graph.pose_noise])
        else: pose = self.graph.pose_eye[None].repeat(len(self.train_data),1,1)
        # use Embedding so it could be checkpointed
        self.graph.optimised_training_poses = torch.nn.Embedding(len(self.train_data),12,_weight=pose.view(-1,12)).to(opt.device)

        # auto near/far for blender dataset
        if opt.data.dataset=="blender":
            idx_range = torch.arange(len(self.train_data),dtype=torch.long,device=opt.device)
            idx_X,idx_Y = torch.meshgrid(idx_range,idx_range)
            self.graph.idx_grid = torch.stack([idx_X,idx_Y],dim=-1).view(-1,2)

        if opt.error_map_size:
            self.graph.error_map = torch.ones([len(self.train_data), opt.error_map_size*opt.error_map_size], dtype=torch.float).to(opt.device)

    def setup_optimizer(self,opt):
        super().setup_optimizer(opt)
        optimizer = getattr(torch.optim,opt.optim.algo)
        self.optim_pose = optimizer([dict(params=self.graph.warp_embedding.parameters(),lr=opt.optim.lr_pose)])
        self.optim_pose.add_param_group(dict(params=self.graph.warp_mlp.parameters(),lr=opt.optim.lr_pose))
        # set up scheduler
        if opt.optim.sched_pose:
            scheduler = getattr(torch.optim.lr_scheduler,opt.optim.sched_pose.type)
            if opt.optim.lr_pose_end:
                assert(opt.optim.sched_pose.type=="ExponentialLR")
                opt.optim.sched_pose.gamma = (opt.optim.lr_pose_end/opt.optim.lr_pose)**(1./opt.max_iter)
            kwargs = { k:v for k,v in opt.optim.sched_pose.items() if k!="type" }
            self.sched_pose = scheduler(self.optim_pose,**kwargs)

    def train_iteration(self,opt,var,loader):
        self.optim_pose.zero_grad()
        if opt.optim.warmup_pose:
            # simple linear warmup of pose learning rate
            self.optim_pose.param_groups[0]["lr_orig"] = self.optim_pose.param_groups[0]["lr"] # cache the original learning rate
            self.optim_pose.param_groups[0]["lr"] *= min(1,self.it/opt.optim.warmup_pose)
        loss = super().train_iteration(opt,var,loader)
        self.optim_pose.step()
        if opt.optim.warmup_pose:
            self.optim_pose.param_groups[0]["lr"] = self.optim_pose.param_groups[0]["lr_orig"] # reset learning rate
        if opt.optim.sched_pose: self.sched_pose.step()
        self.graph.nerf.progress.data.fill_(self.it/opt.max_iter)
        if opt.nerf.fine_sampling:
            self.graph.nerf_fine.progress.data.fill_(self.it/opt.max_iter)
        return loss

    @torch.no_grad()
    def validate(self,opt,ep=None):
        pose,pose_GT = self.get_all_training_poses(opt)
        _,self.graph.sim3 = self.prealign_cameras(opt,pose,pose_GT)
        super().validate(opt,ep=ep)

    @torch.no_grad()
    def log_scalars(self,opt,var,loss,metric=None,step=0,split="train"):
        super().log_scalars(opt,var,loss,metric=metric,step=step,split=split)
        if split=="train":
            # log learning rate
            lr = self.optim_pose.param_groups[0]["lr"]
            self.tb.add_scalar("{0}/{1}".format(split,"lr_pose"),lr,step)
        # compute pose error
        if split=="train" and opt.data.dataset in ["blender","llff"]:
            pose,pose_GT = self.get_all_training_poses(opt)
            pose_aligned,_ = self.prealign_cameras(opt,pose,pose_GT)
            error = self.evaluate_camera_alignment(opt,pose_aligned,pose_GT)
            self.tb.add_scalar("{0}/error_R".format(split),error.R.mean(),step)
            self.tb.add_scalar("{0}/error_t".format(split),error.t.mean(),step)

    @torch.no_grad()
    def visualize(self,opt,var,step=0,split="train"):
        super().visualize(opt,var,step=step,split=split)
        if opt.visdom:
            if split=="val":
                pose,pose_GT = self.get_all_training_poses(opt)
                pose_aligned,_ = self.prealign_cameras(opt,pose,pose_GT)
                util_vis.vis_cameras(opt,self.vis,step=step,poses=[pose_aligned,pose_GT])

    @torch.no_grad()
    def get_all_training_poses(self,opt):
        # get ground-truth (canonical) camera poses
        pose_GT = self.train_data.get_all_camera_poses(opt).to(opt.device)
        pose = self.graph.optimised_training_poses.weight.data.detach().clone().view(-1,3,4)
        return pose,pose_GT

    @torch.no_grad()
    def prealign_cameras(self,opt,pose,pose_GT):
        # compute 3D similarity transform via Procrustes analysis
        center = torch.zeros(1,1,3,device=opt.device)
        center_pred = camera.cam2world(center,pose)[:,0] # [N,3]
        center_GT = camera.cam2world(center,pose_GT)[:,0] # [N,3]
        try:
            sim3 = camera.procrustes_analysis(center_GT,center_pred)
        except:
            print("warning: SVD did not converge...")
            sim3 = edict(t0=0,t1=0,s0=1,s1=1,R=torch.eye(3,device=opt.device))
        # align the camera poses
        center_aligned = (center_pred-sim3.t1)/sim3.s1@sim3.R.t()*sim3.s0+sim3.t0
        R_aligned = pose[...,:3]@sim3.R.t()
        t_aligned = (-R_aligned@center_aligned[...,None])[...,0]
        pose_aligned = camera.pose(R=R_aligned,t=t_aligned)
        return pose_aligned,sim3

    @torch.no_grad()
    def evaluate_camera_alignment(self,opt,pose_aligned,pose_GT):
        # measure errors in rotation and translation
        R_aligned,t_aligned = pose_aligned.split([3,1],dim=-1)
        R_GT,t_GT = pose_GT.split([3,1],dim=-1)
        R_error = camera.rotation_distance(R_aligned,R_GT)
        t_error = (t_aligned-t_GT)[...,0].norm(dim=-1)
        error = edict(R=R_error,t=t_error)
        return error

    @torch.no_grad()
    def evaluate_full(self,opt):
        self.graph.eval()
        # evaluate rotation/translation
        pose,pose_GT = self.get_all_training_poses(opt)
        pose_aligned,self.graph.sim3 = self.prealign_cameras(opt,pose,pose_GT)
        error = self.evaluate_camera_alignment(opt,pose_aligned,pose_GT)
        print("--------------------------")
        print("rot:   {:8.3f}".format(np.rad2deg(error.R.mean().cpu())))
        print("trans: {:10.5f}".format(error.t.mean()))
        print("--------------------------")
        # dump numbers
        quant_fname = "{}/quant_pose.txt".format(opt.output_path)
        with open(quant_fname,"w") as file:
            for i,(err_R,err_t) in enumerate(zip(error.R,error.t)):
                file.write("{} {} {}\n".format(i,err_R.item(),err_t.item()))
        # evaluate novel view synthesis
        super().evaluate_full(opt)

    @torch.enable_grad()
    def evaluate_test_time_photometric_optim(self,opt,var):
        # use another se3 Parameter to absorb the remaining pose errors
        var.se3_refine_test = torch.nn.Parameter(torch.zeros(1,6,device=opt.device))
        optimizer = getattr(torch.optim,opt.optim.algo)
        optim_pose = optimizer([dict(params=[var.se3_refine_test],lr=opt.optim.lr_pose)])
        iterator = tqdm.trange(opt.optim.test_iter,desc="test-time optim.",leave=False,position=1)
        for it in iterator:
            optim_pose.zero_grad()
            var.pose_refine_test = camera.lie.se3_to_SE3(var.se3_refine_test)
            var = self.graph.forward(opt,var,mode="test-optim")
            loss = self.graph.compute_loss(opt,var,mode="test-optim")
            loss = self.summarize_loss(opt,var,loss)
            loss.all.backward()
            optim_pose.step()
            iterator.set_postfix(loss="{:.3f}".format(loss.all))
        return var

    @torch.no_grad()
    def generate_videos_pose(self,opt):
        self.graph.eval()
        fig = plt.figure(figsize=(10,10) if opt.data.dataset=="blender" else (16,8))
        cam_path = "{}/poses".format(opt.output_path)
        os.makedirs(cam_path,exist_ok=True)
        ep_list = []
        for ep in range(0,opt.max_iter+1,opt.freq.ckpt):
            # load checkpoint (0 is random init)
            if ep!=0:
                try: util.restore_checkpoint(opt,self,resume=ep)
                except: continue
            # get the camera poses
            pose,pose_ref = self.get_all_training_poses(opt)
            if opt.data.dataset in ["blender","llff"]:
                pose_aligned,_ = self.prealign_cameras(opt,pose,pose_ref)
                pose_aligned,pose_ref = pose_aligned.detach().cpu(),pose_ref.detach().cpu()
                dict(
                    blender=util_vis.plot_save_poses_blender,
                    llff=util_vis.plot_save_poses,
                )[opt.data.dataset](opt,fig,pose_aligned,pose_ref=pose_ref,path=cam_path,ep=ep)
            else:
                pose = pose.detach().cpu()
                util_vis.plot_save_poses(opt,fig,pose,pose_ref=None,path=cam_path,ep=ep)
            ep_list.append(ep)
        plt.close()
        # write videos
        print("writing videos...")
        list_fname = "{}/temp.list".format(cam_path)
        with open(list_fname,"w") as file:
            for ep in ep_list: file.write("file {}.png\n".format(ep))
        cam_vid_fname = "{}/poses.mp4".format(opt.output_path)
        os.system("ffmpeg -y -r 4 -f concat -i {0} -pix_fmt yuv420p {1} >/dev/null 2>&1".format(list_fname,cam_vid_fname))
        os.remove(list_fname)

# ============================ computation graph for forward/backprop ============================

class Graph(nerf.Graph):

    def __init__(self,opt):
        super().__init__(opt)
        self.nerf = NeRF(opt)
        if opt.nerf.fine_sampling:
            self.nerf_fine = NeRF(opt)
        self.pose_eye = torch.eye(3,4).to(opt.device)

    def get_pose(self,opt,var,mode=None):
        if mode=="train":
            # add the pre-generated pose perturbations
            if opt.data.dataset=="blender":
                if opt.camera.noise:
                    var.pose_noise = self.pose_noise[var.idx]
                    pose = camera.pose.compose([var.pose, var.pose_noise])
                else: pose = var.pose
            else: pose = self.pose_eye[None]
            # add learnable pose correction
            batch_size = len(var.idx)

            if opt.error_map_size:
                num_points = var.ray_idx.shape[1]
                camera_cords_grid_3D = camera.gather_camera_cords_grid_3D(opt,batch_size,intr=var.intr,ray_idx=var.ray_idx).detach()
            else:
                num_points = len(var.ray_idx)
                camera_cords_grid_3D = camera.get_camera_cords_grid_3D(opt,batch_size,intr=var.intr,ray_idx=var.ray_idx).detach()

            camera_cords_grid_2D = camera_cords_grid_3D[...,:2]
            embedding = self.warp_embedding.weight[var.idx,None,:].expand(-1,num_points,-1)
            local_se3_refine = self.warp_mlp(opt,torch.cat((camera_cords_grid_2D,embedding),dim=-1))
            local_pose_refine = camera.lie.se3_to_SE3(local_se3_refine)
            local_pose = camera.pose.compose([local_pose_refine, pose[:,None,...]])
            return local_pose

        elif mode in ["val","eval","test-optim"]:
            # align test pose to refined coordinate system (up to sim3)
            sim3 = self.sim3
            center = torch.zeros(1,1,3,device=opt.device)
            center = camera.cam2world(center,var.pose)[:,0] # [N,3]
            center_aligned = (center-sim3.t0)/sim3.s0@sim3.R*sim3.s1+sim3.t1
            R_aligned = var.pose[...,:3]@self.sim3.R
            t_aligned = (-R_aligned@center_aligned[...,None])[...,0]
            pose = camera.pose(R=R_aligned,t=t_aligned)
            if opt.optim.test_photo and mode!="val":
                pose = camera.pose.compose([var.pose_refine_test, pose])

        else: pose = var.pose
        return pose

    def forward(self,opt,var,mode=None):
        # rescale the size of the scene condition on the pose
        if opt.data.dataset=="blender":
            depth_min,depth_max = opt.nerf.depth.range
            position = camera.Pose().invert(self.optimised_training_poses.weight.data.detach().clone().view(-1,3,4))[...,-1]
            diameter = ((position[self.idx_grid[...,0]]-position[self.idx_grid[...,1]]).norm(dim=-1)).max()
            depth_min_new = (depth_min/(depth_max+depth_min))*diameter
            depth_max_new = (depth_max/(depth_max+depth_min))*diameter
            opt.nerf.depth.range = [depth_min_new, depth_max_new]

        # render images
        batch_size = len(var.idx)
        if opt.nerf.rand_rays and mode in ["train"]:
            # sample rays for optimization
            if opt.error_map_size:
                sample_weight = self.error_map + 2*self.error_map.mean(-1,keepdim=True) # 1/3 importance + 2/3 random
                var.ray_idx_coarse = torch.multinomial(sample_weight, opt.nerf.rand_rays//batch_size, replacement=False) # [B, N], but in [0, opt.error_map_size*opt.error_map_size)
                inds_x, inds_y = var.ray_idx_coarse // opt.error_map_size, var.ray_idx_coarse % opt.error_map_size # `//` will throw a warning in torch 1.10... anyway.
                sx, sy = opt.H / opt.error_map_size, opt.W / opt.error_map_size
                inds_x = (inds_x * sx + torch.rand(batch_size, opt.nerf.rand_rays//batch_size, device=opt.device) * sx).long().clamp(max=opt.H - 1)
                inds_y = (inds_y * sy + torch.rand(batch_size, opt.nerf.rand_rays//batch_size, device=opt.device) * sy).long().clamp(max=opt.W - 1)
                var.ray_idx = inds_x * opt.W + inds_y
            else:
                var.ray_idx = torch.randperm(opt.H*opt.W,device=opt.device)[:opt.nerf.rand_rays//batch_size]# 3/3 random

            local_pose = self.get_pose(opt,var,mode=mode)
            ret = self.local_render(opt,local_pose,intr=var.intr,ray_idx=var.ray_idx,mode=mode) # [B,N,3],[B,N,1]
        elif opt.nerf.rand_rays and mode in ["test-optim"]:
            # sample random rays for optimization
            var.ray_idx = torch.randperm(opt.H*opt.W,device=opt.device)[:opt.nerf.rand_rays//batch_size]
            pose = self.get_pose(opt,var,mode=mode)
            ret = self.render(opt,pose,intr=var.intr,ray_idx=var.ray_idx,mode=mode) # [B,N,3],[B,N,1]
        else:
            # render full image (process in slices)
            pose = self.get_pose(opt,var,mode=mode)
            ret = self.render_by_slices(opt,pose,intr=var.intr,mode=mode) if opt.nerf.rand_rays else \
                  self.render(opt,pose,intr=var.intr,mode=mode) # [B,HW,3],[B,HW,1]
        var.update(ret)
        return var

    def compute_loss(self,opt,var,mode=None):
        loss = edict()
        batch_size = len(var.idx)
        image = var.image.view(batch_size,3,opt.H*opt.W).permute(0,2,1)
        if opt.nerf.rand_rays and mode in ["train","test-optim"]:
            if opt.error_map_size:
                image = torch.gather(image, 1, var.ray_idx[...,None].expand(-1,-1,3))
            else:
                image = image[:,var.ray_idx]

        # compute image losses
        if opt.loss_weight.render is not None:
            render_error = ((var.rgb-image)**2).mean(-1)
            loss.render = render_error.mean()
            if mode in ["train"] and opt.error_map_size:
                ema_error = 0.1 * torch.gather(self.error_map, 1, var.ray_idx_coarse) + 0.9 * render_error.detach()
                self.error_map.scatter_(1, var.ray_idx_coarse, ema_error)

        if opt.loss_weight.render_fine is not None:
            assert(opt.nerf.fine_sampling)
            loss.render_fine = self.MSE_loss(var.rgb_fine,image)

        # global alignment
        if mode in ["train"]:
            source = torch.cat((var.camera_grid_3D,var.camera_center),dim=1)
            target = torch.cat((var.grid_3D,var.center),dim=1)
            R_global, t_global = roma.rigid_points_registration(target, source)
            svd_poses = torch.cat((R_global,t_global[...,None]),-1)
            self.optimised_training_poses.weight.data = svd_poses.detach().clone().view(-1,12)
            if opt.loss_weight.global_alignment is not None:
                loss.global_alignment = self.MSE_loss(target,camera.cam2world(source,svd_poses))

        return loss

    def local_render(self,opt,local_pose,intr=None,ray_idx=None,mode=None):
        batch_size = len(local_pose)
        if opt.error_map_size:
            camera_grid_3D = camera.gather_camera_cords_grid_3D(opt,batch_size,intr=intr,ray_idx=ray_idx).detach()
        else:
            camera_grid_3D = camera.get_camera_cords_grid_3D(opt,batch_size,intr=intr,ray_idx=ray_idx).detach()

        camera_center = torch.zeros_like(camera_grid_3D) # [B,HW,3]
        grid_3D = camera.cam2world(camera_grid_3D[...,None,:],local_pose)[...,0,:] # [B,HW,3]
        center = camera.cam2world(camera_center[...,None,:],local_pose)[...,0,:] # [B,HW,3]
        ray = grid_3D-center # [B,HW,3]
        ret = edict(camera_grid_3D=camera_grid_3D, camera_center=camera_center, grid_3D=grid_3D, center=center) # [B,HW,K] use for global alignment
        if opt.camera.ndc:
            # convert center/ray representations to NDC
            center,ray = camera.convert_NDC(opt,center,ray,intr=intr)
        # render with main MLP
        depth_samples = self.sample_depth(opt,batch_size,num_rays=ray.shape[1]) # [B,HW,N,1]
        rgb_samples,density_samples = self.nerf.forward_samples(opt,center,ray,depth_samples,mode=mode)
        rgb,depth,opacity,prob = self.nerf.composite(opt,ray,rgb_samples,density_samples,depth_samples)
        ret.update(rgb=rgb,depth=depth,opacity=opacity) # [B,HW,K]
        # render with fine MLP from coarse MLP
        if opt.nerf.fine_sampling:
            with torch.no_grad():
                # resample depth acoording to coarse empirical distribution
                depth_samples_fine = self.sample_depth_from_pdf(opt,pdf=prob[...,0]) # [B,HW,Nf,1]
                depth_samples = torch.cat([depth_samples,depth_samples_fine],dim=2) # [B,HW,N+Nf,1]
                depth_samples = depth_samples.sort(dim=2).values
            rgb_samples,density_samples = self.nerf_fine.forward_samples(opt,center,ray,depth_samples,mode=mode)
            rgb_fine,depth_fine,opacity_fine,_ = self.nerf_fine.composite(opt,ray,rgb_samples,density_samples,depth_samples)
            ret.update(rgb_fine=rgb_fine,depth_fine=depth_fine,opacity_fine=opacity_fine) # [B,HW,K]
        return ret

class NeRF(nerf.NeRF):

    def __init__(self,opt):
        super().__init__(opt)
        self.progress = torch.nn.Parameter(torch.tensor(0.)) # use Parameter so it could be checkpointed

    def positional_encoding(self,opt,input,L): # [B,...,N]
        input_enc = super().positional_encoding(opt,input,L=L) # [B,...,2NL]
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