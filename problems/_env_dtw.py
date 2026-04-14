from problems import  VRPTW_Environment
import torch

class DVRPTW_Environment(VRPTW_Environment):
    CUST_FEAT_SIZE = 7

    def _keep_alive_until_reveal(self, acted_veh_idx, cust_idx):
        # If depot is selected while hidden customers still exist, keep at least
        # one vehicle active and advance time to the next reveal instant.
        unrevealed = self.cust_mask[:, 1:].any(dim = 1, keepdim = True)
        force_wait = (cust_idx == 0) & unrevealed
        if not force_wait.any():
            return

        hidden = self.cust_mask.clone()
        hidden[:, 0] = False
        hidden_times = self.nodes[:, :, 6].masked_fill(~hidden, float("inf"))
        next_reveal = hidden_times.min(dim = 1, keepdim = True).values

        inst_wait = force_wait.squeeze(1)
        if inst_wait.any():
            next_t = next_reveal[inst_wait]
            self.vehicles[inst_wait, :, 3] = torch.maximum(self.vehicles[inst_wait, :, 3], next_t)

        self.veh_done.scatter_(1, acted_veh_idx, torch.zeros_like(cust_idx, dtype = torch.bool))
        self.done = bool(self.veh_done.all())
        self._update_cur_veh()

    # Cập nhật khách mới reveal tại thời điểm
    def _update_hidden(self):
        time = self.cur_veh[:, :, 3].clone()
        if self.init_cust_mask is None:
            reveal = self.cust_mask & (self.nodes[:,:,6] <= time)
        else:
            reveal = (self.cust_mask ^ self.init_cust_mask) & (self.nodes[:,:,6] < time)
        if reveal.any():
            self.new_customers = True
            self.cust_mask = self.cust_mask ^ reveal
            self.mask = self.mask ^ reveal[:,None,:].expand(-1,self.veh_count,-1)
            self.veh_done = self.veh_done & (reveal.any(1) ^ True).unsqueeze(1)
            self.vehicles[:, :, 3] = torch.max(self.vehicles[:, :, 3], time)
            self._update_cur_veh()

    def reset(self):
        self.vehicles = self.nodes.new_zeros((self.minibatch_size, self.veh_count, self.VEH_STATE_SIZE))
        self.vehicles[:,:,:2] = self.nodes[:,0:1,:2]
        self.vehicles[:,:,2] = self.veh_capa

        self.veh_done = self.nodes.new_zeros((self.minibatch_size, self.veh_count), dtype = torch.bool)
        self.done = False

        self.cust_mask = (self.nodes[:,:,6] > 0)
        if self.init_cust_mask is not None:
            self.cust_mask = self.cust_mask | self.init_cust_mask
        self.new_customers = True
        self.served = torch.zeros_like(self.cust_mask)

        self.mask = self.cust_mask[:,None,:].repeat(1, self.veh_count, 1)

        self.cur_veh_idx = self.nodes.new_zeros((self.minibatch_size, 1), dtype = torch.int64)
        self.cur_veh = self.vehicles.gather(1, self.cur_veh_idx[:,:,None].expand(-1,-1,self.VEH_STATE_SIZE))
        self.cur_veh_mask = self.mask.gather(1, self.cur_veh_idx[:,:,None].expand(-1,-1,self.nodes_count))

    # nhận vào action của xe hiện tại
    # Trả về reward
    def step(self, cust_idx):
        acted_veh_idx = self.cur_veh_idx.clone()
        reward = super().step(cust_idx)
        self._keep_alive_until_reveal(acted_veh_idx, cust_idx)
        self._update_hidden()
        return reward
