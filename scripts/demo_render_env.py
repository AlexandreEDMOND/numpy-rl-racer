from numpy_rl_racer.env import Obstacle, RacingEnv
from numpy_rl_racer.rendering import MatplotlibRenderer


def main():
    obstacles = [
        Obstacle(x=1.5, y=-2.0, radius=0.5),
        Obstacle(x=-1.0, y=1.5, radius=0.4),
        Obstacle(x=3.0, y=2.0, radius=0.45),
    ]
    env = RacingEnv(obstacles=obstacles)
    env.reset(seed=0)
    renderer = MatplotlibRenderer(env.track)

    actions = [
        [0.0, 3.0],
        [0.3, 2.0],
        [-0.2, 2.0],
        [0.0, 1.0],
        [0.5, 2.0],
    ]

    print("Running demo with obstacles...")
    for i, action in enumerate(actions):
        obs, reward, done, _ = env.step(action)
        print(f"  obs dim={len(obs)}")
        renderer.render(env.state, step=i + 1, reward=reward, obstacles=env.obstacles)
        print(f"  step={i + 1:2d}  x={env.state.x:6.2f}  y={env.state.y:6.2f}  "
              f"heading={env.state.heading:5.2f}  reward={reward:5.2f}")
        if done:
            print(f"Episode finished at step {i + 1}")
            break

    print("Demo complete. Close the plot window to exit.")
    renderer.show()


if __name__ == "__main__":
    main()
