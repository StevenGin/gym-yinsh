"""
Play a random game and save every N-th frame as a PNG, plus a final board image.

    python scripts/save_game_frames.py --out /tmp/yinsh_frames --every 5

Useful when a display is not available (CI, remote server).
"""
import argparse
import os
import numpy as np
from PIL import Image
import gymnasium as gym
import gym_yinsh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/tmp/yinsh_frames")
    parser.add_argument("--every", type=int, default=10, help="Save every N steps")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    env = gym.make("yinsh-v0", render_mode="rgb_array")
    obs, info = env.reset(seed=args.seed)
    rng = np.random.default_rng(args.seed)

    step = 0
    frame = 0
    while True:
        if step % args.every == 0:
            img = env.render()
            path = os.path.join(args.out, f"frame_{frame:04d}_step{step:04d}.png")
            Image.fromarray(img).save(path)
            frame += 1
            print(f"  saved {path}")

        mask = info["action_mask"]
        valid = np.where(mask)[0]
        if len(valid) == 0:
            break
        action = rng.choice(valid)
        obs, reward, terminated, truncated, info = env.step(action)
        step += 1

        if terminated or truncated:
            img = env.render()
            path = os.path.join(args.out, f"frame_{frame:04d}_final.png")
            Image.fromarray(img).save(path)
            print(f"  saved final: {path}")
            print(f"Game over after {step} steps. Winner: {info['winner']}")
            break

    env.close()


if __name__ == "__main__":
    main()
