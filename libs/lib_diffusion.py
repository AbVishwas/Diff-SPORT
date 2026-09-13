import numpy as np
import torch
import torch.nn.functional as F
from libs.lib_svd import Inpainting_custom, Inpainting_custom_mask_batches

# The numbering of timesteps starts from 1 to T and for index 0, alpha=1, beta=0
def forward_diffusion(x, t_curr, t_next, schedule): #only for visualization (works with numpy inputs)
    
    #INPUT ASSERTIONS: BEGIN
    assert isinstance(x, np.ndarray)
    assert isinstance(t_curr, int)
    assert isinstance(t_next, int)
    assert isinstance(schedule, dict)
    assert 'alpha' in schedule and 'beta' in schedule
    alpha = schedule.get('alpha')
    beta  = schedule.get('beta')
    assert isinstance(alpha, (list, np.ndarray)) and isinstance(beta, (list, np.ndarray))
    assert (len(alpha) == len(beta))
    assert (alpha[0] == 1.0) and (beta[0] == 0)
    T = len(alpha) - 1
    #assert T > 10, f"Horizon T={T} is too low!"
    assert (t_curr >=0) and (t_curr <= T)
    assert t_next >= t_curr
    assert t_next <= T
    #INPUT ASSERTIONS: END

    #ALGORITHM: BEGIN
    alpha_effective = alpha[t_next]/alpha[t_curr]
    variance_factor = 1.0 - alpha_effective
    std_factor = np.sqrt(variance_factor)
    mean = np.sqrt(alpha_effective) * x
    return mean + (std_factor*np.random.normal(loc=0.0, scale=1.0, size=x.shape))
    #ALGORITHM: END

def reverse_diffusion(ddpm_model, x, t_curr, schedule, cuda=True, onlymean=False): #only for visualization (works with numpy inputs)

    #INPUT ASSERTIONS: BEGIN
    assert isinstance(x, np.ndarray)
    assert isinstance(t_curr, int)
    assert isinstance(schedule, dict)
    assert 'alpha' in schedule and 'beta' in schedule and 'rvar' in schedule
    alpha = schedule.get('alpha')
    beta  = schedule.get('beta')
    rvar = schedule.get('rvar')
    assert isinstance(alpha, (list, np.ndarray)) and isinstance(beta, (list, np.ndarray)) and isinstance(rvar, (list, np.ndarray))
    assert (len(alpha) == len(beta)) and (len(alpha) == len(rvar))
    assert (alpha[0] == 1.0) and (beta[0] == 0) and (rvar[0] == 0)
    T = len(alpha) - 1
    assert (t_curr > 0) and (t_curr <= T)
    #INPUT ASSERTIONS: END

    #ALGORITHM: BEGIN  
    x_inp = torch.from_numpy(x)
    t_curr_inp = t_curr * torch.ones(x_inp.shape[0], dtype=torch.float32)
    device = 'cuda' if cuda else 'cpu'
    device = torch.device(device)
    ddpm_model.to_device(device) # model is an object of DDPM class below
    #x_inp = x_inp.to(device)
    #t_curr_inp = t_curr_inp.to(device)
    with torch.no_grad():
        noise_pred = ddpm_model.infer(x_inp, t_curr_inp)
        noise_pred = noise_pred.cpu().numpy()
    assert noise_pred.shape == x.shape and noise_pred.dtype == x.dtype
    mean = (1.0/np.sqrt(1.0-beta[t_curr])) * ( x - ( (beta[t_curr]/np.sqrt(1-alpha[t_curr])) * noise_pred ) )
    assert mean.dtype == x.dtype
    rstd = np.sqrt(rvar[t_curr])
    if onlymean:
        return mean
    
    return mean + np.asarray((rstd*np.random.normal(loc=0.0, scale=1.0, size=tuple(x.shape))),dtype=np.float32)
    #ALGORITHM: END

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return betas

def make_schedule(scheme, rvar, start_beta=0, end_beta=0.99, T=500):
    
    #INPUT ASSERTIONS: BEGIN
    assert isinstance(T, int)
    assert T > 0
    #assert T > 10, f"Horizon T={T} is too low!"
    assert scheme=='linear' or scheme == 'cosine'
    assert rvar=='beta' or rvar=='fvar'
    assert end_beta >= start_beta
    #INPUT ASSERTIONS: END

    if scheme == 'linear':
        ans = 1.0
        alpha = [1.0,]
        eps = 1e-7
        beta = [0,] + np.linspace(start=start_beta, stop=end_beta, endpoint=True, num=T).tolist()
        beta = np.array(beta, dtype=np.float32)
        assert (len(beta) == T+1) and (beta[0] == 0) and (beta[T] - end_beta < eps) and (beta[1] - start_beta < eps)
        for i in range(1,len(beta)):
                ans *= 1.0-beta[i]
                alpha.append(ans)
        alpha = np.array(alpha, dtype=np.float32)
    
    elif scheme == 'cosine':
        import math
        ans = 1.0
        alpha = [1.0,]
        eps = 1e-7
        beta = [0,] + betas_for_alpha_bar(T, lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,)
        beta = np.array(beta, dtype=np.float32)
        assert (len(beta) == T+1) and (beta[0] == 0)
        for i in range(1,len(beta)):
                ans *= 1.0-beta[i]
                alpha.append(ans)
        alpha = np.array(alpha, dtype=np.float32)

    else:
        raise NotImplementedError
    
    if rvar == 'beta':

        rvar = np.copy(beta)

    elif rvar == 'fvar':

        rvar = [0.0]
        for i in range(1,len(beta)):
            rvar.append( ((1-alpha[i-1])*beta[i])/(1-alpha[i]) )
        rvar = np.array(rvar, dtype=np.float32)     

    else:
        raise NotImplementedError
    
    assert (len(alpha) == len(beta)) and (len(beta) == len(rvar))

    return {'alpha':alpha, 'beta':beta, 'rvar':rvar, 'T':T}

def sample_timesteps(T_ddpm, T_short=None, sigmasquare=None):

    #Sample a DDRM subsequence of time steps from DDPM schedule

    #INPUT ASSERTIONS: BEGIN
    assert isinstance(T_ddpm, int)
    assert not (T_short is None and sigmasquare is None)
    if T_short is not None:
        assert isinstance(T_short, int)
        assert T_short <= T_ddpm and T_short >= 1
    #INPUT ASSERTIONS: END

    if T_short is not None:
        #ALGORITHM: BEGIN
        T_short_list = [T_ddpm,]
        s = T_ddpm // T_short    
        while len(T_short_list) < T_short:
                t = T_short_list[0] - s
                t = t if t >= 1 else 0
                T_short_list = [t,] + T_short_list
        T_short_list = np.asarray(T_short_list)
        assert len(T_short_list) == T_short
        return T_short_list
        #ALGORITHM: END  
    
    else:
        # make adaptive schedule: optimized for sigmasquare_{t} = 2*sigmasquare_{t-1}
        #ALGORITHM: BEGIN
        assert len(sigmasquare)-1 == T_ddpm
        T_short_list = [T_ddpm,]
        pivot = sigmasquare[T_ddpm]    
        rev_T_list = np.arange(1,T_ddpm+1)[::-1]
        rev_sigmasquare_list = sigmasquare[1:][::-1]
        for t_,s_ in zip(rev_T_list, rev_sigmasquare_list):
            if s_ <= pivot/2:
                T_short_list = [t_,] + T_short_list
                pivot /= 2
                #pivot = T_short_list[0]    
        return np.asarray(T_short_list)
        #ALGORITHM: END  
    

class DDPM():

    def __init__(self, schedule, model, weightedloss=True, cuda=True):
        
        #INPUT ASSERTIONS: BEGIN
        assert isinstance(schedule, dict)
        assert 'alpha' in schedule and 'beta' in schedule and 'rvar' in schedule
        alpha = schedule.get('alpha')
        beta  = schedule.get('beta')
        rvar = schedule.get('rvar')
        assert isinstance(alpha, (list, np.ndarray)) and isinstance(beta, (list, np.ndarray)) and isinstance(rvar, (list, np.ndarray))
        assert (len(alpha) == len(beta)) and (len(alpha) == len(rvar))
        assert (alpha[0] == 1.0) and (beta[0] == 0) and (rvar[0] == 0)
        #INPUT ASSERTIONS: END

        #super(DDPM).__init__()
        self.T = len(alpha) - 1
        self.alpha = alpha
        self.beta = beta
        self.rvar = rvar
        self.weightedloss = weightedloss
        #assert weightedloss==False, "weightedloss not supported at the moment!"
        device = 'cuda' if cuda else 'cpu'
        self.device = torch.device(device)    
        self.model = model.to(self.device)
        self.loss = self.__build_loss().to(self.device)
        self.loss_weights = self.__get_loss_weights()
    
    
    def __build_loss(self,):
        return torch.nn.MSELoss(reduction='none')

    def __get_loss_weights(self,):
            
            t_index = np.arange(start=2, stop=1+self.T, step=1)
            numer = .5 * np.square(self.beta[t_index])
            denom = self.rvar[t_index]*(1.0 - self.beta[t_index])*(1.0-self.alpha[t_index])
            loss_weights = numer/(denom+1e-10)
            loss_weights = np.asarray([0,0,] + loss_weights.tolist(), dtype=np.float32)
            return loss_weights


    def __sample_std_gaussian_noise(self, X): # X is already supposed to be on self.device

        return torch.randn(*X.shape, device=self.device, dtype=torch.float32)
    
    def __generate_temporal_noise(self, X0): # X0 here is supposed to be already on self.device

        num_samples = X0.shape[0]
        
        #SAMPLE T
        start_time_step = 1
        if self.weightedloss:
            start_time_step = 2 # This is to ensure non-inf weight for t=1 case
        T_curr = np.random.choice(np.arange(start=start_time_step, stop=1+self.T, step=1), size=num_samples, replace=True)    
        Alpha_T_curr = self.alpha[T_curr]
        
        #GENERATE WEIGHTS FOR LOSS
        if self.weightedloss:
            loss_weights = self.loss_weights[T_curr]
            loss_weights = torch.from_numpy(loss_weights).to(self.device)
        else:
            loss_weights = 1.0

        #GENERATE X_T
        p = torch.tensor(np.sqrt(Alpha_T_curr).reshape(-1, *([1]*(len(X0.shape)-1)) ), dtype=torch.float32).to(self.device)
        q = torch.tensor(np.sqrt(1.0-Alpha_T_curr).reshape(-1, *([1]*(len(X0.shape)-1)) ), dtype=torch.float32).to(self.device)
        Noise_std_gaussian = self.__sample_std_gaussian_noise(X0)
        X_T_curr = (p*X0) + (q*Noise_std_gaussian)
        T_curr = torch.from_numpy(np.asarray(T_curr, dtype=np.float32)).to(self.device)

        return X_T_curr, T_curr, Noise_std_gaussian, loss_weights

    def to_device(self, device):
        self.device = device
        self.model = self.model.to(self.device)
        self.loss = self.loss.to(self.device)

    def infer(self, X_T_curr, T_curr, mode="eval", y_c=None): #only for inference
        if mode=="eval":
            self.model.eval()
        else:
            self.model.train()
        X_T_curr, T_curr = X_T_curr.to(self.device), T_curr.to(self.device)
        Noise_std_gaussian_pred = self.model(X_T_curr, T_curr, y=y_c)
        
        return Noise_std_gaussian_pred

    
    def run_step(self, X0): #only for training/validation run
        
        self.model.train()
        X0 = X0.to(self.device)
        X_T_curr, T_curr, Noise_std_gaussian, loss_weights = self.__generate_temporal_noise(X0)
        Noise_std_gaussian_pred = self.model(X_T_curr, T_curr)
        loss = torch.mean(self.loss(Noise_std_gaussian_pred, Noise_std_gaussian).reshape(X0.shape[0],-1).mean(dim=-1) * loss_weights)
        return loss
    
    
    def ddim_step(self, x, noise_pred, alpha_current, alpha_prev, eta):
        
        if eta is not None:
            sigmasquare_prev = (eta**2) * ((1. - alpha_prev)/(1. - alpha_current)) * (1. - (alpha_current/alpha_prev))
        else:
            sigmasquare_prev = 1. - alpha_prev 
        mean = float(np.sqrt(alpha_prev/alpha_current)) * (x - float(np.sqrt(1. - alpha_current))*noise_pred) + float(np.sqrt(1. - alpha_prev - sigmasquare_prev)) * noise_pred
        std = float(np.sqrt(sigmasquare_prev)) * torch.randn_like(mean)
        return (mean, std)
        
    def generate_sample_batch(self, sample_batch_shape, sampler, T_sub, eta, onlymean):
        assert sampler in ['ddim','itdn']  #ddim, iterative denoiser. for ddim one needs to specify eta, if eta=1 -> ddpm sampler, eta=0 -> deterministic sampler
        assert (T_sub > 0 and T_sub <= self.T), f"T_sub and self.T are:{T_sub, self.T}"
        if sampler == 'ddim':
            assert (eta >=0 and eta <= 1.)
        else:
            eta = None # this enforces itdn update in ddim_step 
        
        T_sub_list = sample_timesteps(self.T, T_short=T_sub)
        T_sub_list = T_sub_list[::-1]
        T_sub_next_list = np.concatenate([T_sub_list[1:],[0]])

        x = torch.randn(*sample_batch_shape, device=self.device, dtype=torch.float32)
        spare_ones = torch.ones(x.shape[0], dtype=torch.float32, device=x.device)
                
        for t_curr, t_prev in zip(T_sub_list, T_sub_next_list):
                        #print(f"{t_curr}-->{t_prev}")
                        t_curr_inp = t_curr * spare_ones
                        
                        with torch.no_grad():
                            noise_pred = self.infer(x, t_curr_inp)
                        assert noise_pred.shape == x.shape and noise_pred.dtype == x.dtype
                        mean, std = self.ddim_step(x, noise_pred, self.alpha[t_curr], self.alpha[t_prev], eta)
                        if onlymean:
                            std = 0.
                        x = torch.as_tensor(mean + std, dtype=x.dtype)
        
        return x

    def generate_samples(self, num_samples=0, sample_batch_size=None, sample_shape=None, sampler=None, T_sub=None, eta=None, onlymean=False): #only for generation

        assert num_samples > 0
        assert sample_batch_size > 0
        assert isinstance(sample_shape, tuple)
        # sample shape = shape of single sample [c, dim1, dim2, dim3,...]

        alpha = self.alpha 
        beta  = self.beta 
        rvar =  self.rvar 
        T = len(alpha) - 1

        num_iter = num_samples // sample_batch_size
        num_generated_samples = 0
        generated_samples = []

        while num_generated_samples < num_samples:
            
            bs = (num_samples - num_generated_samples) // sample_batch_size
            bs = sample_batch_size if bs >= 1 else (num_samples - num_generated_samples)
            sample_batch_shape = (bs,) + sample_shape

            generated_sample_batch = self.generate_sample_batch(sample_batch_shape=sample_batch_shape, sampler=sampler, T_sub=T_sub, eta=eta, onlymean=onlymean) 
            
            num_generated_samples += generated_sample_batch.shape[0]
            generated_samples.append(generated_sample_batch.cpu().numpy())
            print(f"{num_generated_samples} samples generated")
            
        generated_samples = np.concatenate(generated_samples, axis=0)
        assert generated_samples.shape == (num_samples, *sample_shape)
        return generated_samples

#------------------------------------------------------ Lib utils for DDRM/PGDM ------------------------------------------


class DDPM_R(DDPM):
    
    def __init__(self, schedule, model, weightedloss=True, cuda=True,):
        
        super().__init__(schedule, model, weightedloss, cuda)
        self.sigmasquare = (1.0/(self.alpha+1e-12)) - 1.0   
    
    def Denoiser(self, x, t, t_tensor, y_c=None):

            eps_t_pred = self.infer(torch.as_tensor(np.sqrt(self.alpha[t])*x,dtype=torch.float32), t_tensor, mode="eval", y_c=y_c)
            eps_t_pred = eps_t_pred[:,:3] # model also predicts sigma values so 6 channels in total, for mean we only need first 3 channels
            if torch.any(torch.isnan(eps_t_pred)):
                        eps_t_pred[torch.isnan(eps_t_pred)] = 0
                        print(f"In Denoiser:: eps_t_pred has a nan")
                
            x0 = x - np.sqrt(self.sigmasquare[t])*(eps_t_pred)
            return x0

    def reverse_diffusion_gd(self, x_shape, y, sigma_y, H=None, T_ddrm=20, cuda=True, onlymean=False, 
                             karras=False, adaptive=False, y_c=None, class_cond=None, gditer=0, 
                             seed=0, numproc=1, procnum=0, idx_so_far=0): #only for generation

                #INPUT ASSERTIONS AND SETUP: BEGIN
                assert len(x_shape) == 4 # this is a random tensor of (batch, c, h, w)
                assert len(y.shape) == 2
                device = 'cuda' if cuda else 'cpu'
                device = torch.device(device)
                assert H is not None
                T = len(self.alpha) - 1
                
                if not karras:
                    if not adaptive:
                        T_ddrm_list = sample_timesteps(T, T_short=T_ddrm)
                    else:
                        T_ddrm_list = sample_timesteps(T, T_short=None, sigmasquare=self.sigmasquare)
                else:
                    T_ddrm_list = sample_timesteps(T, T_short=T_ddrm, sigmasquare=None, karras=True)

                assert (y_c is None) == (class_cond is False)
                #INPUT ASSERTIONS AND SETUP: END


                #PREPROCESSING: BEGIN  
                #----------------------------------------------------SVD based pre-processing BEGIN---------------------------------
                y = y.to(device)
                #dummy_x = dummy_x.to(device)
                b, m = y.shape
                fulldimx = x_shape[1]*x_shape[2]*x_shape[3]
                n = fulldimx
                
                #--------------------------------------------
                singulars = torch.zeros((n),device=y.device)
                _singulars = H.singulars() 
                singulars[:_singulars.shape[0]] = _singulars 
                assert singulars.shape[0] == n       
                k = int(torch.sum(singulars > 0))
                
                #--------------------------------------------
                S_inv_diag = torch.zeros((1,n),device=y.device)
                S_inv_diag[:,singulars > 0] = (1.0 / singulars)[singulars > 0]
                
                S_diag = torch.zeros((1,n),device=y.device)
                S_diag[:,singulars > 0] = singulars[singulars > 0]
                
                S_mask = torch.zeros((1,n),device=y.device)
                S_mask[:,singulars > 0] = 1.0
                assert k == int(torch.sum(S_mask))

                S_mask_bool = S_mask > 0
                S_mask_bool = S_mask_bool.reshape((1,n))
                mask_singular = ~S_mask_bool
                #--------------------------------------------
                U_t_y = H.Ut(y) 
                _y_bar =  U_t_y * S_inv_diag[:,:U_t_y.shape[-1]]  #/ singulars[:U_t_y.shape[-1]]
                y_bar = torch.zeros((b,n),device=y.device)
                y_bar[:,:_y_bar.shape[1]] = _y_bar
                assert y_bar.shape == (b, n)
                
                #----------------------------------------------------SVD based pre-processing END---------------------------------
                #PREPROCESSING: END
                
                #ALGORITHM: BEGIN                
                gens = [torch.Generator(device=y.device).manual_seed(seed +  procnum + (idx_so_far+i)*numproc  ) for i in range(b)]

                if onlymean:
                    x_bar_T = torch.zeros_like(y_bar, dtype=torch.float32) #experimental
                else:
                    x_bar_T = float(np.sqrt(self.sigmasquare[T])) * torch.cat([torch.randn((1, *x_shape[1:]), generator=gens[i], dtype=y.dtype, device=y.device) for i in range(b)], dim=0,)    
                    x_bar_T = x_bar_T.reshape((b,-1))
                    
                x_inp_bar = x_bar_T
                x_inp = H.V(x_inp_bar)
                
                T_ddrm_list = T_ddrm_list[::-1]
                T_ddrm_next_list = np.concatenate([T_ddrm_list[1:],[0]])
                
                spare_ones = torch.ones(x_inp.shape[0], dtype=torch.float32, device=x_inp.device)
                
                for t_curr, t_prev in zip(T_ddrm_list, T_ddrm_next_list):
                        
                        t_curr_inp = t_curr * spare_ones
                        x_inp = x_inp.reshape(x_shape).detach()
                        lr = 1.
                        #print(f"t = {t_curr}")
                        no_prior = False
                        if no_prior:
                            print(f"no prior set to True")
                        
                        surrogate_prior = False
                        if not no_prior and surrogate_prior:
                            print(f"using one-step surrogate prior")
                        

                        for iternum in range(gditer):
                            
                                x_ = x_inp.detach()
                                x_.requires_grad = True    
                                
                                eps_t_pred = self.infer(torch.as_tensor(np.sqrt(self.alpha[t_curr])*x_,dtype=torch.float32), t_curr_inp, mode="eval", y_c=y_c)
                                
                                if torch.any(torch.isnan(eps_t_pred)):
                                        print(f"t={t_curr}, eps_t_pred has a nan")
                                        eps_t_pred[torch.isnan(eps_t_pred)] = 0.
                                        
                                mu = x_ - np.sqrt(self.sigmasquare[t_curr])*(eps_t_pred)
                                
                                with torch.no_grad():
                                
                                    
                                    x_0 = mu.detach()
                                    if not no_prior:
                                        if not surrogate_prior:
                                            eps_0_pred = self.infer(torch.as_tensor(np.sqrt(self.alpha[1])*x_0,dtype=torch.float32), 1*spare_ones, mode="eval", y_c=y_c)
                                            #lam_wt = 1.
                                        else:
                                            z_prev = t_prev//10
                                            if z_prev == 0:
                                                 z_prev = 1
                                            x_t_prev_surrogate = x_0 + np.sqrt(self.sigmasquare[z_prev]) * torch.cat([torch.randn((1, *x_0.shape[1:]), generator=gens[i], dtype=x_0.dtype, device=x_0.device) for i in range(b)], dim=0,)
                                            eps_0_pred = self.infer(torch.as_tensor(np.sqrt(self.alpha[z_prev])*x_t_prev_surrogate, dtype=torch.float32), z_prev*spare_ones, mode="eval", y_c=y_c)
                                            #lam_wt = (self.sigmasquare[1] / self.sigmasquare[z_prev]) #if t_prev > 0 else 0.
                                        
                                        if torch.any(torch.isnan(eps_0_pred)):
                                            print(f"t={t_curr}, eps_0_pred has a nan")
                                            eps_0_pred[torch.isnan(eps_0_pred)] = 0.

                                        if not surrogate_prior:    
                                            Dx_0 = x_0 - np.sqrt(self.sigmasquare[1])*(eps_0_pred)
                                        else:
                                            Dx_0 = x_0 - np.sqrt(self.sigmasquare[z_prev])*(eps_0_pred)
                                    
                                    vec1 = 0 if no_prior else (Dx_0 - x_0).reshape((x_shape[0],-1))
                                    vec2 = (y_bar - x_0.reshape((x_shape[0],-1))) * S_mask 
                                    vec = vec1 + vec2

                                
                                
                                mu = mu.reshape((x_shape[0],-1))
                                loss_ =  torch.sum(mu * vec.detach())
                                x_grad = torch.autograd.grad(loss_, x_,create_graph=False, retain_graph=False)[0]
                                
                                if torch.any(torch.isnan(x_grad)):
                                    print(f"NaN in xgrad in guidance")
                                    x_grad[torch.isnan(x_grad)] = 0.
                                
                                edm_y_grad = x_grad.detach()
                                #print(f"x_grad = {torch.sum(edm_y_grad**2, dim=[1,2,3])}")
                                x_inp = x_inp + lr * edm_y_grad
                                
                                if iternum == gditer-1:

                                    with torch.no_grad():
                                        
                                        eps_t_pred = self.infer(torch.as_tensor(np.sqrt(self.alpha[t_curr])*x_inp.detach(),dtype=torch.float32), t_curr_inp, mode="eval", y_c=y_c)
                                        
                                        if torch.any(torch.isnan(eps_t_pred)):
                                            print(f"t={t_curr}, eps_t_pred has a nan in last block")
                                            eps_t_pred[torch.isnan(eps_t_pred)] = 0.
                                            
                                        mu = x_inp.detach() - np.sqrt(self.sigmasquare[t_curr])*(eps_t_pred)
                                        mu = mu.reshape((x_shape[0],-1))
                                        loss_ = torch.sum( S_mask*(mu - y_bar)**2, dim=-1)
                                        print(f"iter = {iternum}, loss = {torch.mean(loss_)}", flush=True)


                        mu = mu.detach()
                        mu = mu.reshape((x_shape[0],-1))
                        y_bar = y_bar.to(mu.dtype)    
                        if sigma_y == 0 and ( t_prev == 0 ):
                            #print(f"only do the correct update at the end")
                            mu[:,S_mask_bool[0]] = y_bar[:,S_mask_bool[0]]
                        
                        mu = mu.reshape(x_shape)
                        std = np.sqrt(self.sigmasquare[t_prev])
                        x_inp = mu + std * torch.cat([torch.randn((1, *mu.shape[1:]), generator=gens[i], dtype=mu.dtype, device=mu.device) for i in range(b)], dim=0,)
                        
                return x_inp            
                #ALGORITHM: END





    def reverse_diffusion_gd_mask_batches(self, x_shape, y, sigma_y, H=None, T_ddrm=20, cuda=True, onlymean=False, 
                             karras=False, adaptive=False, y_c=None, class_cond=None, gditer=0, 
                             seed=0, numproc=1, procnum=0, idx_so_far=0): #only for generation

                #INPUT ASSERTIONS AND SETUP: BEGIN
                assert len(x_shape) == 4 # this is a random tensor of (batch, c, h, w)
                assert len(y.shape) == 2
                device = 'cuda' if cuda else 'cpu' #TODO: Get Global device
                device = torch.device(device)
                assert H is not None, "You must provide a batch of H corresponding to y"

                T = len(self.alpha) - 1
                
                if not karras:
                    if not adaptive:
                        T_ddrm_list = sample_timesteps(T, T_short=T_ddrm)
                    else:
                        T_ddrm_list = sample_timesteps(T, T_short=None, sigmasquare=self.sigmasquare)
                else:
                    T_ddrm_list = sample_timesteps(T, T_short=T_ddrm, sigmasquare=None, karras=True)

                assert (y_c is None) == (class_cond is False)
                #INPUT ASSERTIONS AND SETUP: END


                #PREPROCESSING: BEGIN  
                #----------------------------------------------------SVD based pre-processing BEGIN---------------------------------
                y = y.to(device)
                #dummy_x = dummy_x.to(device)
                b, m = y.shape
                fulldimx = x_shape[1]*x_shape[2]*x_shape[3]
                n = fulldimx
                
                #--------------------------------------------
                # Handle batch-wise masks
                #H = Inpainting_custom_mask_batches(mask_tensor=masks, device=device)
                singulars = H.singulars()
                assert singulars.shape == (b, n), "Singular values shape mismatch"     
                k = int(torch.sum(singulars > 0))
                
                #--------------------------------------------
                S_inv_diag = torch.zeros((b, n), device=y.device)
                S_inv_diag[singulars > 0] = (1.0 / singulars[singulars > 0])
                
                S_diag = torch.zeros((b, n), device=y.device)
                S_diag[singulars > 0] = singulars[singulars > 0]
                
                S_mask = torch.zeros((b, n), device=y.device)
                S_mask[singulars > 0] = 1.0
                
                assert k == int(torch.sum(S_mask))

                S_mask_bool = S_mask > 0
                #--------------------------------------------
                U_t_y = H.Ut(y) 
                y_bar =  U_t_y * S_inv_diag

                assert y_bar.shape == (b, n)
                
                #----------------------------------------------------SVD based pre-processing END---------------------------------
                #PREPROCESSING: END
                
                #ALGORITHM: BEGIN                
                gens = [torch.Generator(device=y.device).manual_seed(seed +  procnum + (idx_so_far+i)*numproc  ) for i in range(b)]

                if onlymean:
                    x_bar_T = torch.zeros_like(y_bar, dtype=torch.float32) #experimental #TODO:get global datatype
                else:
                    x_bar_T = float(np.sqrt(self.sigmasquare[T])) * torch.cat([torch.randn((1, *x_shape[1:]), generator=gens[i], dtype=y.dtype, device=y.device) for i in range(b)], dim=0,)    
                    x_bar_T = x_bar_T.reshape((b,-1))
                    
                x_inp_bar = x_bar_T
                x_inp = H.V(x_inp_bar)
                
                T_ddrm_list = T_ddrm_list[::-1]
                T_ddrm_next_list = np.concatenate([T_ddrm_list[1:],[0]])
                
                spare_ones = torch.ones(x_inp.shape[0], dtype=torch.float32, device=x_inp.device) #TODO:get global datatype
                
                for t_curr, t_prev in zip(T_ddrm_list, T_ddrm_next_list):
                        
                        t_curr_inp = t_curr * spare_ones
                        x_inp = x_inp.reshape(x_shape).detach()
                        lr = 1.   #TODO: Check
                        #print(f"t = {t_curr}")
                        no_prior = False
                        if no_prior:
                            print(f"no prior set to True")
                        
                        surrogate_prior = False
                        if not no_prior and surrogate_prior:
                            print(f"using one-step surrogate prior")
                        

                        for iternum in range(gditer):
                            
                                x_ = x_inp.detach()
                                x_.requires_grad = True    
                                
                                eps_t_pred = self.infer(torch.as_tensor(np.sqrt(self.alpha[t_curr])*x_,dtype=torch.float32), t_curr_inp, mode="eval", y_c=y_c)
                                
                                if torch.any(torch.isnan(eps_t_pred)):
                                        print(f"t={t_curr}, eps_t_pred has a nan")
                                        eps_t_pred[torch.isnan(eps_t_pred)] = 0.
                                        
                                mu = x_ - np.sqrt(self.sigmasquare[t_curr])*(eps_t_pred)
                                
                                with torch.no_grad():
                                
                                    
                                    x_0 = mu.detach()
                                    if not no_prior:
                                        if not surrogate_prior:
                                            eps_0_pred = self.infer(torch.as_tensor(np.sqrt(self.alpha[1])*x_0,dtype=torch.float32), 1*spare_ones, mode="eval", y_c=y_c)
                                            #lam_wt = 1.
                                        else:
                                            z_prev = t_prev//10
                                            if z_prev == 0:
                                                 z_prev = 1
                                            x_t_prev_surrogate = x_0 + np.sqrt(self.sigmasquare[z_prev]) * torch.cat([torch.randn((1, *x_0.shape[1:]), generator=gens[i], dtype=x_0.dtype, device=x_0.device) for i in range(b)], dim=0,)
                                            eps_0_pred = self.infer(torch.as_tensor(np.sqrt(self.alpha[z_prev])*x_t_prev_surrogate, dtype=torch.float32), z_prev*spare_ones, mode="eval", y_c=y_c)
                                            #lam_wt = (self.sigmasquare[1] / self.sigmasquare[z_prev]) #if t_prev > 0 else 0.
                                        
                                        if torch.any(torch.isnan(eps_0_pred)):
                                            print(f"t={t_curr}, eps_0_pred has a nan")
                                            eps_0_pred[torch.isnan(eps_0_pred)] = 0.

                                        if not surrogate_prior:    
                                            Dx_0 = x_0 - np.sqrt(self.sigmasquare[1])*(eps_0_pred)
                                        else:
                                            Dx_0 = x_0 - np.sqrt(self.sigmasquare[z_prev])*(eps_0_pred)
                                    
                                    vec1 = 0 if no_prior else (Dx_0 - x_0).reshape((x_shape[0],-1))
                                    vec2 = (y_bar - x_0.reshape((x_shape[0],-1))) * S_mask 
                                    vec = vec1 + vec2

                                
                                
                                mu = mu.reshape((x_shape[0],-1))
                                loss_ =  torch.sum(mu * vec.detach())
                                x_grad = torch.autograd.grad(loss_, x_,create_graph=False, retain_graph=False)[0]
                                
                                if torch.any(torch.isnan(x_grad)):
                                    print(f"NaN in xgrad in guidance")
                                    x_grad[torch.isnan(x_grad)] = 0.
                                
                                edm_y_grad = x_grad.detach()
                                #print(f"x_grad = {torch.sum(edm_y_grad**2, dim=[1,2,3])}")
                                x_inp = x_inp + lr * edm_y_grad
                                
                                if iternum == gditer-1:

                                    with torch.no_grad():
                                        
                                        eps_t_pred = self.infer(torch.as_tensor(np.sqrt(self.alpha[t_curr])*x_inp.detach(),dtype=torch.float32), t_curr_inp, mode="eval", y_c=y_c)
                                        
                                        if torch.any(torch.isnan(eps_t_pred)):
                                            print(f"t={t_curr}, eps_t_pred has a nan in last block")
                                            eps_t_pred[torch.isnan(eps_t_pred)] = 0.
                                            
                                        mu = x_inp.detach() - np.sqrt(self.sigmasquare[t_curr])*(eps_t_pred)
                                        mu = mu.reshape((x_shape[0],-1))
                                        loss_ = torch.sum( S_mask*(mu - y_bar)**2, dim=-1)
                                        #print(f"iter = {iternum}, loss = {torch.mean(loss_)}")


                        mu = mu.detach()
                        mu = mu.reshape((x_shape[0],-1))
                  
                        y_bar = y_bar.to(mu.dtype)    
                        if sigma_y == 0 and ( t_prev == 0 ):
                            #print(f"only do the correct update at the end")
                            mu[S_mask_bool] = y_bar[S_mask_bool]
                        
                        mu = mu.reshape(x_shape)
                        std = np.sqrt(self.sigmasquare[t_prev])
                        x_inp = mu + std * torch.cat([torch.randn((1, *mu.shape[1:]), generator=gens[i], dtype=mu.dtype, device=mu.device) for i in range(b)], dim=0,)
                        
                return x_inp            
                #ALGORITHM: END


    def reverse_diffusion_ddrm(self, x_shape, y, sigma_y, H=None, T_ddrm=20, cuda=True, onlymean=False, adaptive=False, y_c=None, class_cond=None): 

        with torch.no_grad():
                
                #INPUT ASSERTIONS AND SETUP: BEGIN
                assert len(y.shape) == 2
                assert sigma_y == 0
                device = 'cuda' if cuda else 'cpu'
                device = torch.device(device)
                assert H is not None
                T = len(self.alpha) - 1
                if not adaptive:
                    T_ddrm_list = sample_timesteps(T, T_short=T_ddrm)
                else:
                    T_ddrm_list = sample_timesteps(T, T_short=None, sigmasquare=self.sigmasquare)
                assert (y_c is None) == (class_cond is False)
                #INPUT ASSERTIONS AND SETUP: END

                #PREPROCESSING: BEGIN  
                #----------------------------------------------------SVD based pre-processing BEGIN---------------------------------
                y = y.to(device)
                b, m = y.shape
                b_, n = x_shape[0], np.prod(x_shape[1:])
                assert b_ == b and n == m
                #--------------------------------------------
                
                singulars = H.singulars() 
                assert singulars.shape == (n,)       
                k = int(torch.sum(singulars > 0))
                #--------------------------------------------
                
                S_mask = torch.zeros((1,n),device=y.device)
                S_mask[:,singulars > 0] = 1.0
                assert k == int(torch.sum(S_mask))

                S_mask_bool = S_mask > 0
                S_mask_bool = S_mask_bool.reshape((1,n))
                
                #----------------------------------------------------SVD based pre-processing END---------------------------------
                #PREPROCESSING: END
                
                #ALGORITHM: BEGIN                
                if onlymean:
                    x_T = torch.zeros_like(y, dtype=torch.float32)
                else:
                    std = np.sqrt(self.sigmasquare[T])
                    x_T = std * torch.randn(*y.shape, dtype=torch.float32, device=y.device)
                        
                x_inp = x_T.reshape(x_shape)

                T_ddrm_list = T_ddrm_list[::-1]
                T_ddrm_next_list = np.concatenate([T_ddrm_list[1:],[0]])
                
                spare_ones = torch.ones(b, dtype=torch.float32, device=y.device)

                for t_curr, t_prev in zip(T_ddrm_list, T_ddrm_next_list):
                    
                        t_curr_inp = t_curr * spare_ones
                        
                        x0_pred = self.Denoiser(x_inp, t_curr, t_curr_inp, y_c=y_c)
                        assert x0_pred.shape == x_shape
                        
                        mean = x0_pred.reshape((b,n)) ;y = y.to(mean.dtype)    
                        mean[:,S_mask_bool[0]] = y[:,S_mask_bool[0]]
                        std = np.sqrt(self.sigmasquare[t_prev])
                        
                        if not onlymean:
                            x_inp = mean + std * torch.randn(*mean.shape, device=mean.device, dtype=torch.float32)
                        else:
                            x_inp = mean

                        x_inp = x_inp.reshape(x_shape)

                return x_inp            
                #ALGORITHM: END

    def reverse_diffusion_pgdm(self, x_shape, y, sigma_y, H=None, T_ddrm=20, cuda=True, onlymean=False, adaptive=False, y_c=None, class_cond=None): #only for generation

                
                #INPUT ASSERTIONS AND SETUP: BEGIN
                assert len(y.shape) == 2
                assert sigma_y == 0
                device = 'cuda' if cuda else 'cpu'
                device = torch.device(device)
                assert H is not None
                T = len(self.alpha) - 1
                if not adaptive:
                    T_ddrm_list = sample_timesteps(T, T_short=T_ddrm)
                else:
                    T_ddrm_list = sample_timesteps(T, T_short=None, sigmasquare=self.sigmasquare)
                assert (y_c is None) == (class_cond is False)
                #INPUT ASSERTIONS AND SETUP: END

                #PREPROCESSING: BEGIN  
                #----------------------------------------------------SVD based pre-processing BEGIN---------------------------------
                y = y.to(device)
                b, m = y.shape
                b_, n = x_shape[0], np.prod(x_shape[1:])
                assert b_ == b and n == m
                #--------------------------------------------
                
                singulars = H.singulars() 
                assert singulars.shape == (n,)       
                k = int(torch.sum(singulars > 0))
                #--------------------------------------------
                
                S_mask = torch.zeros((1,n),device=y.device)
                S_mask[:,singulars > 0] = 1.0
                assert k == int(torch.sum(S_mask))

                S_mask_bool = S_mask > 0
                S_mask_bool = S_mask_bool.reshape((1,n))
                
                #----------------------------------------------------SVD based pre-processing END---------------------------------
                #PREPROCESSING: END
                
                #ALGORITHM: BEGIN                
                if onlymean:
                    x_T = torch.zeros_like(y, dtype=torch.float32)
                else:
                    std = np.sqrt(self.sigmasquare[T])
                    x_T = std * torch.randn(*y.shape, dtype=torch.float32, device=y.device)
                        
                x_inp = x_T.reshape(x_shape)

                T_ddrm_list = T_ddrm_list[::-1]
                T_ddrm_next_list = np.concatenate([T_ddrm_list[1:],[0]])
                
                spare_ones = torch.ones(b, dtype=torch.float32, device=y.device)
                
                for t_curr, t_prev in zip(T_ddrm_list, T_ddrm_next_list):
                        
                        t_curr_inp = t_curr * spare_ones
                        x_inp = x_inp.detach()
                        x_inp.requires_grad=True
                        
                        x0_pred = self.Denoiser(x_inp, t_curr, t_curr_inp, y_c=y_c)
                        x0_pred_bar = x0_pred.reshape(b,n)
                        
                        res = (y - x0_pred_bar)*S_mask
                        eps_loss_singular = torch.sum(res**2)
                        eps_loss_singular.backward()
                        grad_x = x_inp.grad.detach()

                        if torch.any(torch.isnan(grad_x)):
                            print(f"grad_x has a nan")
                            grad_x[torch.isnan(grad_x)] = 0.
                        
                        mean = (x0_pred.detach() - (0.5 * grad_x)).reshape(b,n) ;y = y.to(mean.dtype) 
                        mean[:, S_mask_bool[0]] = y[:,S_mask_bool[0]]
                        std = np.sqrt(self.sigmasquare[t_prev])
                        
                        if not onlymean:
                            x_inp = mean + std * torch.randn(*mean.shape, device=mean.device, dtype=torch.float32)
                        else:
                            x_inp = mean

                        #######################x_inp_bar[:, S_mask_bool[0]] = y_bar[:,S_mask_bool[0]]
                        x_inp = x_inp.reshape(x_shape).detach()  
                        
                return x_inp            

