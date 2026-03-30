from problems import VRPTW_Environment
import torch

class SVRPTW_Environment(VRPTW_Environment):
    def __init__(self, data, nodes = None, cust_mask = None,
            pending_cost = 2, late_cost = 1,
            speed_var = 0.1, late_p = 0.05, slow_down = 0.5, late_var = 0.3):
        """
            Initializes the environment with parameters controlling speed, lateness, and costs.

            Args:
                data: Input data required for the environment (type depends on parent class).
                nodes (optional): List or array of nodes to consider. Defaults to None.
                cust_mask (optional): Mask or filter for customers. Defaults to None.
                pending_cost (float, optional): Cost incurred for each pending task. Defaults to 2.
                late_cost (float, optional): Cost incurred for each late task. Defaults to 1.
                speed_var (float, optional): Variability in speed; higher values mean more fluctuation in speed. Defaults to 0.1.
                late_p (float, optional): Probability of a task being late. Defaults to 0.05.
                slow_down (float, optional): Factor by which speed is reduced in certain conditions. Defaults to 0.5.
                late_var (float, optional): Variability in lateness; higher values mean more fluctuation in lateness. Defaults to 0.3.

            Variables Explained:
                - speed_var: Độ biến thiên của tốc độ di chuyển, giá trị càng lớn thì tốc độ càng không ổn định.
                - late_p: Xác suất một nhiệm vụ bị trễ.
                - slow_down: Hệ số làm chậm tốc độ khi gặp điều kiện bất lợi.
                - late_var: Độ biến thiên của thời gian trễ, giá trị càng lớn thì thời gian trễ càng không ổn định.
                - pending_cost: Chi phí cho mỗi nhiệm vụ đang chờ xử lý.
                - late_cost: Chi phí cho mỗi nhiệm vụ bị trễ.
            """
        super().__init__(data, nodes, cust_mask, pending_cost, late_cost)
        self.speed_var = speed_var
        self.late_p = late_p
        self.slow_down = slow_down
        self.late_var = late_var

    def _sample_speed(self):
        late = self.nodes.new_empty((self.minibatch_size, 1)).bernoulli_(self.late_p)
        rand = torch.randn_like(late)
        speed = late * self.slow_down * (1 + self.late_var * rand) + (1-late) * (1 + self.speed_var * rand)
        return speed.clamp_(min = 0.1) * self.veh_speed
