import torch
import math
from trajdata.augmentation.augmentation import BatchAugmentation
from trajdata.data_structures.batch import AgentBatch, SceneBatch
from trajdata.utils.arr_utils import PadDirection, mask_up_to

def propagate_control_noise(state_tensor, sigma_acc=0.1, sigma_omega=0.05, delta_t=0.1):
    """
    Propagate control noise through the state components assuming integrator dynamics.
    state_tensor shape: [batch_size, sequence_length, feature_dim]
    feature_dim order: "x, y, z, xd, yd, xdd, ydd, s, c"
    """
    if state_tensor.dim()==3:
        state_tensor = state_tensor.unsqueeze(1)
    B, A, T, _ = state_tensor.shape

    batch_size, seq_len, _ = state_tensor.shape

    # Generate noise for acceleration and turning rate
    acc_noise = torch.normal(0, sigma_acc, (B, A, T))
    omega_noise = torch.normal(0, sigma_omega, (B,A, T))

    # Update heading angle phi based on omega noise
    phi = torch.atan2(state_tensor[..., 7], state_tensor[..., 8])  # Extract phi from s and c
    phi_new = phi + omega_noise * delta_t
    
    # Compute new s and c based on updated phi
    s_new = torch.sin(phi_new)
    c_new = torch.cos(phi_new)
    
    new_tensor = state_tensor.clone()
    
    # Update accelerations (xdd, ydd) based on acc noise and direction
    new_tensor[..., 5] += acc_noise * c_new  # xdd = acc * cos(phi)
    new_tensor[..., 6] += acc_noise * s_new  # ydd = acc * sin(phi)
    
    # Update velocities (xd, yd) based on accelerations and delta_t
    new_tensor[..., 3] += state_tensor[..., 5] * delta_t  # xd += xdd * delta_t
    new_tensor[..., 4] += state_tensor[..., 6] * delta_t  # yd += ydd * delta_t
    
    # Update positions (x, y) based on velocities and delta_t
    new_tensor[..., 0] += state_tensor[..., 3] * delta_t  # x += xd * delta_t
    new_tensor[..., 1] += state_tensor[..., 4] * delta_t  # y += yd * delta_t
    
    # Update s and c in the state tensor
    new_tensor[..., 7] = s_new
    new_tensor[..., 8] = c_new

    #only return the noise term for diffusion schedule
    return new_tensor-state_tensor
def scale_normalize(
    x,
    # mean = torch.tensor([1.8592e-02, 8.8420e-06, 1.7937e+00, -5.6197e-04, 1.6397e+00, 9.2974e-06]) ,
    # std = torch.tensor([0.9935, 0.0552,0.0,4.1853, 0.3312, 3.3478, 0.0614])
    mean = torch.tensor([1.8592e-02, 8.8420e-06,0.0, 1.7937,-0.0001008,1.6397,-0.000921,0.0,0.0]),
    std  = torch.tensor([0.9935, 0.0552,1e-12,4.1853,0.5941,3.3478,0.5431,0.3312,0.3312])
    ):
    #check shape 
    assert x.shape[-1] == mean.shape[-1]
    #expand mean and std to x shape
    mean = mean.expand(x.shape)
    std = std.expand(x.shape)
    return (x - mean) / std
def scale_unnormalize(
    x,
    # mean = torch.tensor([1.8592e-02, 8.8420e-06, 1.7937e+00, -5.6197e-04, 1.6397e+00, 9.2974e-06]) ,
    # std = torch.tensor([0.9935, 0.0552,0.0,4.1853, 0.3312, 3.3478, 0.0614])
    mean = torch.tensor([1.8592e-02, 8.8420e-06,0.0, 1.7937,-0.0001008,1.6397,-0.000921,0.0,0.0]),
    std  = torch.tensor([0.9935, 0.0552,0.0,4.1853,0.5941,3.3478,0.5431,0.3312,0.3312])
    ):
    #check shape 
    assert x.shape[-1] == mean.shape[-1]
    return x * std + mean
class VarianceSchedule(torch.nn.Module):

    def __init__(self, num_steps, mode='cosine',beta_1=1e-4, beta_T=5e-2,cosine_s=8e-3):
        super().__init__()
        assert mode in ('linear', 'cosine')
        print(f"diffusion mode: {mode}")
        self.num_steps = num_steps
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.mode = mode

        if mode == 'linear':
            betas = torch.linspace(beta_1, beta_T, steps=num_steps)
        elif mode == 'cosine':
            timesteps = (
            torch.arange(num_steps + 1) / num_steps + cosine_s
            )
            alphas = timesteps / (1 + cosine_s) * math.pi / 2
            alphas = torch.cos(alphas).pow(2)
            alphas = alphas / alphas[0]
            betas = 1 - alphas[1:] / alphas[:-1]
            betas = betas.clamp(max=0.999)

        betas = torch.cat([torch.zeros([1]), betas], dim=0)     # Padding

        alphas = 1 - betas
        log_alphas = torch.log(alphas)
        for i in range(1, log_alphas.size(0)):  # 1 to T
            log_alphas[i] += log_alphas[i - 1]
        alpha_bars = log_alphas.exp()

        sigmas_flex = torch.sqrt(betas)
        sigmas_inflex = torch.zeros_like(sigmas_flex)
        for i in range(1, sigmas_flex.size(0)):
            sigmas_inflex[i] = ((1 - alpha_bars[i-1]) / (1 - alpha_bars[i])) * betas[i]
        sigmas_inflex = torch.sqrt(sigmas_inflex)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('sigmas_flex', sigmas_flex)
        self.register_buffer('sigmas_inflex', sigmas_inflex)

    def uniform_sample_t(self, batch_size):
        ts = np.random.choice(np.arange(1, self.num_steps+1), batch_size)
        return ts.tolist()

    def get_sigmas(self, t, flexibility):
        assert 0 <= flexibility and flexibility <= 1
        sigmas = self.sigmas_flex[t] * flexibility + self.sigmas_inflex[t] * (1 - flexibility)
        return sigmas

    def add_noise(self, x, t=10):
        batch_size, _, point_dim = x.size()
        
        x = scale_normalize(x)
        if t == None:
            t = self.uniform_sample_t(batch_size)

        alpha_bar = self.alpha_bars[t]
        beta = self.betas[t]

        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1)      # (B, 1, 1)
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1)  # (B, 1, 1)

        e_rand = torch.randn_like(x) # (B, N, d)
        e_rand[...,2] = 0.0 #z don't have noise
           
        x_pert = c0 * x +  c1 * e_rand
        
        return scale_unnormalize(x_pert)
class DiffusionNoise(BatchAugmentation):
    def __init__(
        self,
        mean: float = 0.0,
        stddev: float = 0.2,
    ) -> None:
        self.mean = mean
        self.stddev = stddev
        
        self.variance_schedule = VarianceSchedule(num_steps=100, mode='cosine',beta_1=1e-4, beta_T=5e-2,cosine_s=8e-3)

    def apply_agent(self, agent_batch: AgentBatch) -> None:
        
        
        agent_hist_noise = torch.normal(
            self.mean, self.stddev, size=agent_batch.agent_hist.shape
        )
        
        neigh_hist_noise = torch.normal(
            self.mean, self.stddev, size=agent_batch.neigh_hist.shape
        )

        if agent_batch.history_pad_dir == PadDirection.BEFORE:
            agent_hist_noise[..., -1, :] = 0
            neigh_hist_noise[..., -1, :] = 0
        else:
            len_mask = ~mask_up_to(
                agent_batch.agent_hist_len,
                delta=-1,
                max_len=agent_batch.agent_hist.shape[1],
            ).unsqueeze(-1)
            agent_hist_noise[len_mask.expand(-1, -1, agent_hist_noise.shape[-1])] = 0

            len_mask = ~mask_up_to(
                agent_batch.neigh_hist_len,
                delta=-1,
                max_len=agent_batch.neigh_hist.shape[2],
            ).unsqueeze(-1)
            neigh_hist_noise[
                len_mask.expand(-1, -1, -1, neigh_hist_noise.shape[-1])
            ] = 0

        # agent_batch.agent_hist += agent_hist_noise
        agent_batch.agent_hist = self.variance_schedule.add_noise(agent_batch.agent_hist)
        agent_batch.neigh_hist += neigh_hist_noise

    def apply_scene(self, scene_batch: SceneBatch) -> None:
        scene_batch.agent_hist[..., :-1, :] += torch.normal(
            self.mean, self.stddev, size=scene_batch.agent_hist[..., :-1, :].shape
        )
        
