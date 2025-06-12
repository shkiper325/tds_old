from stable_baselines import DQN

import tensorflow as tf

from env import Env

def main():
    env = Env(0)

    policy_kwargs = {'act_fun' : tf.nn.leaky_relu, 'layers' : [32, 32, 32]}
    print(policy_kwargs)

    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        exploration_fraction=0.25,
        buffer_size=10000,
        exploration_final_eps=0.1,
        prioritized_replay=True,
        tensorboard_log='tb.log',
        n_cpu_tf_sess=4,
        learning_rate = 0.0000625,
        batch_size=64,
        policy_kwargs=policy_kwargs,
    )
    model.learn(total_timesteps=1000000)
    model.save("fucking_fuck")

    del model # remove to demonstrate saving and loading

    model = DQN.load("fucking_fuck")

    for _ in range(5):
        obs = env.reset()
        done = False
        while not done:
            action, _state = model.predict(obs)
            obs, reward, done, info = env.step(action)
            env.render()

if __name__ == '__main__':
    main()