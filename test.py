import random
from env import create_train_env, MACRO_NUMS

env = create_train_env(1, 3, True, 'test.mp4')
done = False 

while not done:
    # action = random.randint(0, 35)
    action = random.randint(18, 18 + MACRO_NUMS - 1)
    frames, reward, done, info = env.step(action)
    
    print(frames.shape)
    print(action)
    print(reward)
    print(done)
    print(info['stage'])