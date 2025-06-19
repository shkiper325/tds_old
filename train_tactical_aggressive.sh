#!/bin/bash

# Команда для тактического агрессивного обучения
# Комбинирует агрессивные награды, оборонительные штрафы и фокус на дистанции

echo "🎯 Запуск тактического агрессивного обучения..."
echo "📋 Параметры:"
echo "   • Агрессивная игра: W_KILL = 50.0, W_TICK = -0.005"
echo "   • Оборонительность: W_HIT = -2.0, W_WALL = -1.0"  
echo "   • Фокус на дистанции: W_DIST = 1.0, OPT_DIST = 150"
echo ""

python train.py \
    --episodes 2000 \
    --lr 0.001 \
    --save-interval 200 \
    --gamma 0.99 \
    --entropy-coef 0.01 \
    --hidden-dims 256 128 64 \
    --grad-clip 0.5 \
    --screen-size 400 400 \
    --max-steps 600 \
    --player-speed 450.0 \
    --player-health 3 \
    --reward-damage 1.0 \
    --reward-hit -2.0 \
    --reward-kill 50.0 \
    --reward-tick -0.005 \
    --reward-wall -1.0 \
    --reward-distance 1.0 \
    --optimal-distance 150 \
    --distance-tolerance 100 \
    --pistol-cooldown 250 \
    --shotgun-cooldown 750 \
    --machinegun-cooldown 100 \
    --projectile-speed 300.0 \
    --tensorboard-dir "tb/tactical_aggressive" \
    --checkpoint-dir "checkpoints/tactical_aggressive" \
    --verbose

echo ""
echo "✅ Обучение завершено!"
echo "📊 Для мониторинга запустите: tensorboard --logdir tb/tactical_aggressive"
echo "🎮 Для тестирования: python evaluate.py --agent1 checkpoints/tactical_aggressive/final_model_agent1.pth --hidden-dims 256 128 64"
