from agent import Agent
from game import CarGameAI

def train():
    record = -float("inf")
    agent = Agent()
    game = CarGameAI()
    last_25 = [0]*25
    last_25_max = -float("inf")
    total_rew = 0

    while True:
        
        state_old = agent.get_state(game) # get old state

        final_move = agent.get_action(state_old) # get move

        reward, done = game.play_step(final_move) # perform move
        total_rew += reward #accumulate the obtained reward

        state_new = agent.get_state(game) # get new state

        agent.train_short_memory(state_old, final_move, reward, state_new, done) # train short memory

        agent.remember(state_old, final_move, reward, state_new, done) # remember

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            last_25.insert(0, total_rew)
            last_25.pop()
            last_25_mean = sum(last_25)/len(last_25)

            if total_rew > record:
                record = total_rew

            if last_25_mean > last_25_max:
                last_25_max = last_25_mean
                agent.model.save()

            print('Iter:', agent.n_games, '\t',  'Reward', total_rew, '\t' ,'Max_Seen:', record, '\t', 'Last25Avg:', '{:.2f}'.format(round(last_25_mean, 3)), '\t', 'Last25Max:', '{:.2f}'.format(round(last_25_max, 3)))
            total_rew = 0

if __name__ == '__main__':
    train()