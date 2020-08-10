from simple_spread_test import make_env
from maa2c import MAA2C


if __name__ == '__main__':
	env = make_env(scenario_name="simple_spread",benchmark=False)
	ma_controller = MAA2C(env,False)

	ma_controller.run(10000000,300)
