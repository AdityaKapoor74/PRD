from simple_spread_test import make_env
from maa2c import MAA2C
from torch.utils.tensorboard import SummaryWriter


if __name__ == '__main__':
	env = make_env(scenario_name="simple_spread")
	ma_controller = MAA2C(env)
	# reward_list,entropy_list,value_loss_list,policy_loss_list = ma_controller.run(10,300)
	ma_controller.run(100,300)
	
	# writer = SummaryWriter('runs/simple_spread_lr_2e-4')

	# for i in range(len(ma_controller.agents.entropy_list)):
 #  		writer.add_scalar('Entropy loss',ma_controller.agents.entropy_list[i],i)
 #  		writer.add_scalar('Reward',ma_controller.episode_rewards[i],i)
 #  		writer.add_scalar('Value Loss',ma_controller.agents.value_loss_list[i],i)
 #  		writer.add_scalar('Policy Loss',ma_controller.agents.policy_loss_list[i],i)

	# tensorboard --logdir=runs
