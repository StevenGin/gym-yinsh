"""
Watch a random YINSH game play out in a Pygame window.

    python scripts/watch_random_game.py [--fps 4] [--seed 0]

Controls:
    Escape / close window  →  quit
"""
import argparse
import time
import numpy as np
import gymnasium as gym
import gym_yinsh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fps", type=float, default=4.0, help="Steps per second")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    env = gym.make("yinsh-v0", render_mode="human")
    obs, info = env.reset(seed=args.seed)
    rng = np.random.default_rng(args.seed)

    step_delay = 1.0 / args.fps
    running = True
    step = 0

    while running:
        # render returns False when the window is closed
        result = env.render()
        if result is False:
            break

        if not (obs is None) and not info.get("winner") is not None or True:
            mask = info["action_mask"]
            valid = np.where(mask)[0]
            if len(valid) == 0:
                break
            action = rng.choice(valid)
            obs, reward, terminated, truncated, info = env.step(action)
            step += 1

            if terminated or truncated:
                env.render()  # show final state
                print(f"Game over after {step} steps. Winner: {info['winner']} "
                      f"({'White' if info['winner'] == 0 else 'Black'})")
                # Wait for user to close
                env.unwrapped._pygame_renderer.wait_for_close()
                break

        time.sleep(step_delay)

    env.close()


if __name__ == "__main__":
    main()
