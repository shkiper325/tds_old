from stable_baselines import DQN

import tensorflow as tf

from env import Env

def main():
    env = Env(0)

    policy_kwargs = {'layers' : [32, 32]}
    print(policy_kwargs)

    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        exploration_fraction=0.5,
        buffer_size=1000,
        exploration_final_eps=0.05,
        prioritized_replay=True,
        n_cpu_tf_sess=4,
        learning_rate = 0.0001,
        batch_size=128,
        policy_kwargs=policy_kwargs,
    )
    model.learn(total_timesteps=100000)
    model.save("weird2")

    del model # remove to demonstrate saving and loading

    model = DQN.load("weird2")

    for _ in range(5):
        obs = env.reset()
        done = False
        while not done:
            action, _state = model.predict(obs)
            obs, reward, done, info = env.step(action)
            env.render()

if __name__ == '__main__':
    main()