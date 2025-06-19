# Hyperparameter Tuning Recommendations

Here are some recommended hyperparameter configurations for training your agents.

### 1. Balanced (Good Starting Point)
This configuration provides a good balance between aggressive and tactical behavior.

```bash
python train.py --episodes 20000 --lr 0.001 --gamma 0.99 --entropy-coef 0.01 --hidden-dims 128 128 --save-interval 1000 --reward-kill 20.0 --reward-damage 1.0 --reward-hit -1.0 --reward-tick -0.01
```

### 2. Aggressive Agent
This setup heavily rewards kills and uses a higher player speed to encourage aggressive behavior.

```bash
python train.py --episodes 20000 --lr 0.001 --gamma 0.99 --entropy-coef 0.01 --hidden-dims 128 128 --save-interval 1000 --reward-kill 50.0 --player-speed 600
```

### 3. Tactical/Defensive Agent
This setup encourages maintaining an optimal distance and has a smaller kill reward, promoting a more tactical playstyle.

```bash
python train.py --episodes 20000 --lr 0.001 --gamma 0.99 --entropy-coef 0.01 --hidden-dims 128 128 --save-interval 1000 --reward-kill 10.0 --reward-distance 1.0 --optimal-distance 200
```

### 4. Deeper Network for Complex Behavior
This uses a larger neural network, which might be able to learn more complex strategies. It may require a lower learning rate and more training episodes.

```bash
python train.py --episodes 30000 --lr 0.0005 --gamma 0.99 --entropy-coef 0.01 --hidden-dims 256 256 128 --save-interval 1000
```
