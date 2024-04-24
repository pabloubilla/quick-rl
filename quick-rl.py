import pandas as pandas
import numpy as np
from Agents import MonteCarloG, MonteCarloG_baseline, MonteCarloG_statevalue, DQNAgent
from scipy.signal import savgol_filter
from sklearn.impute import KNNImputer
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
import itertools
from multiprocessing import Pool

# def run_repetitions(rl_method, n_repetitions, n_episodes, smoothing_window, learning_rate, gamma, max_steps):
#     print(f'Running {rl_method}')
#     learning_curve = np.zeros(shape=(n_repetitions, n_episodes))
#     for rep in range(n_repetitions):
#         print(f'Repetition {rep} ' + rl_method)
#         if rl_method == 'MonteCarloG':
#             pi = MonteCarloG(n_questions, n_questions, learning_rate, gamma)
#         elif rl_method == 'MonteCarloG_baseline':
#             pi = MonteCarloG_baseline(n_questions, n_questions, learning_rate, gamma)
#         for e in range(n_episodes):
#             if verbose: print('-'*20)
#             if verbose: print(f'Episode {e}')
#             # initial s is each question is 0
#             s = np.zeros(n_questions)
#             # maybe state can be current prediction
#             s_p = imputer.transform(np.array([s]))[0]

#             # real_s is one random person from training
#             real_s = train.sample().values[0]

#             # fixed real_s from train
#             # real_s = train.iloc[3].values

#             s_list = []
#             r_list = []
#             a_list = []
#             for i in range(m_questions):
#                 # print current state
#                 if verbose: print('-'*10)
#                 if verbose: print(f'Question {i}')
#                 if verbose: print(f'State {s}')
#                 a, p = pi.select_action(s)
#                 # show probability of each action
#                 if verbose: print(f'Act. prob. {p}')
#                 if verbose: print(f'Action {a}')


#                 # env.render()
#                 s_list.append(s_p)

#                 # get what person answered in question a
#                 if real_s[a] == 1:
#                     # update s
#                     s[a] = 1
#                 else:
#                     s[a] = -1
#                 s_p = imputer.transform(np.array([s]))[0]

#                 if False:
#                     # get reward
#                     r = 0
#                 else:
#                     # create copy of s
#                     s_predict = s.copy()
#                     # change 0 for nan
#                     s_predict[s_predict == 0] = np.nan
#                     # change -1 for 0
#                     s_predict[s_predict == -1] = 0
#                     # predict
#                     s_predict = imputer.transform(np.array([s_predict]))[0]
#                     if verbose: print(f'Predict probs {s_predict}')
#                     # round
#                     s_predict = np.round(s_predict)
#                     if verbose: print(f'Predict vals {s_predict}')

#                     # compute error
#                     r = -np.mean(np.abs(s_predict - real_s))
#                     # r = 0
#                     # if s[a] == 1:
#                     #     r = 1

#                 # print predict, real, and reward
                
#                 if verbose: print(f'Real s {real_s}')
#                 if verbose: print(f'Reward {r}')

#                 a_list.append(a)
#                 r_list.append(r)

#                 if i == m_questions-1:
#                     #print(f'done {e}')
#                     pi.update(r_list,a_list,s_list)
#                     learning_curve[rep, e] = np.mean(r_list)
#                 # els
#                 #     s = s_next
#     learning_curve = np.mean(learning_curve, axis=0)
#     learning_curve = savgol_filter(learning_curve, smoothing_window, 1)
#     # learning_curve = smooth(learning_curve, smoothing_window)
#     return learning_curve
                
# def run_repetitions_v2(rl_method, n_repetitions, n_episodes, smoothing_window, learning_rate, gamma, max_steps):
#     learning_curve = np.zeros(shape=(n_repetitions, n_episodes))
#     final_error_list = np.zeros(shape=(n_repetitions, n_episodes))
#     for rep in range(n_repetitions):
#         agent = MonteCarloG_statevalue(n_questions, n_questions, learning_rate, gamma, epsilon = 0.1) 
#         for e in range(n_episodes):
#             if verbose: print('-'*20)
#             if verbose: print(f'Episode {e}')
#             # initial s is each question is 0
#             s = np.zeros(n_questions)
#             # maybe state can be current prediction
#             s_p = imputer.transform(np.array([s]))[0]

#             # real_s is one random person from training
#             real_s = train.sample().values[0]

#             s_list = []
#             r_list = []
#             a_list = []
#             for i in range(m_questions):
#                 # print current state
#                 if verbose: print('-'*10)
#                 if verbose: print(f'Question {i}')
#                 if verbose: print(f'State {s}')
#                 a, p = agent.select_action(s)
#                 # show probability of each action
#                 if verbose: print(f'Act. prob. {p}')
#                 if verbose: print(f'Action {a}')


#                 # env.render()
#                 s_list.append(s_p)

#                 # get what person answered in question a
#                 if real_s[a] == 1:
#                     # update s
#                     s[a] = 1
#                 else:
#                     s[a] = -1
#                 s_p = imputer.transform(np.array([s]))[0]

#                 if i != i:
#                     # get reward
#                     r = 0
#                 else:
#                     # create copy of s
#                     s_predict = s.copy()
#                     # change 0 for nan
#                     s_predict[s_predict == 0] = np.nan
#                     # change -1 for 0
#                     s_predict[s_predict == -1] = 0
#                     # predict
#                     s_predict = imputer.transform(np.array([s_predict]))[0]
#                     if verbose: print(f'Predict probs {s_predict}')
#                     # round
#                     s_predict = np.round(s_predict)
#                     if verbose: print(f'Predict vals {s_predict}')

#                     # compute error
#                     r = -np.mean(np.abs(s_predict - real_s))

#                 # print predict, real, and reward
                
#                 if verbose: print(f'Real s {real_s}')
#                 if verbose: print(f'Reward {r}')

#                 a_list.append(a)
#                 r_list.append(r)

#                 if i == m_questions-1:
#                     #print(f'done {e}')
#                     agent.update(r_list,a_list,s_list)
#                     learning_curve[rep, e] = np.sum(r_list)
#                     final_error_list[rep, e] = -r
#                 # els
#                 #     s = s_next
#     learning_curve = np.mean(learning_curve, axis=0)
#     learning_curve = savgol_filter(learning_curve, smoothing_window, 1)
#     final_error_list = np.mean(final_error_list, axis=0)
#     final_error_list = savgol_filter(final_error_list, smoothing_window, 1)
#     # learning_curve = smooth(learning_curve, smoothing_window)
#     return learning_curve, final_error_list


def compute_td_loss(agent, target_network, states, actions, rewards, next_states, done_flags,
                    gamma=0.99, device='cpu '):

    # convert numpy array to torch tensors
    states = torch.tensor(states, device=device, dtype=torch.float)
    actions = torch.tensor(actions, device=device, dtype=torch.long)
    rewards = torch.tensor(rewards, device=device, dtype=torch.float)
    next_states = torch.tensor(next_states, device=device, dtype=torch.float)
    done_flags = torch.tensor(done_flags.astype('float32'),device=device,dtype=torch.float)

    # get q-values for all actions in current states
    # use agent network
    predicted_qvalues = agent(states)

    # compute q-values for all actions in next states
    # use target network
    predicted_next_qvalues = target_network(next_states)
    
    # select q-values for chosen actions
    predicted_qvalues_for_actions = predicted_qvalues[range(
        len(actions)), actions]

    # compute Qmax(next_states, actions) using predicted next q-values
    next_state_values,_ = torch.max(predicted_next_qvalues, dim=1)

    # compute "target q-values" 
    target_qvalues_for_actions = rewards + gamma * next_state_values * (1-done_flags)

    # mean squared error loss to minimize
    loss = torch.mean((predicted_qvalues_for_actions -
                       target_qvalues_for_actions.detach()) ** 2)

    return loss

def epsilon_schedule(start_eps, end_eps, step, final_step):
    return start_eps + (end_eps-start_eps)*min(step, final_step)/final_step

def greedy_0(imputer, train, T_questions):
    # test error using a greedy 0 policy
    N_questions = train.shape[1]
    error_greedy_list = []
    for person_i in range(train.shape[0]):
        real_s_i = train.iloc[person_i].values
        s_i = np.nan*np.ones(N_questions)
        for j in range(T_questions):
            probs_ij = imputer.transform(np.array([s_i]))[0]
            # compute variance
            var_ij = probs_ij*(1-probs_ij)
            # print(f'Variances {var_ij}')
            # choose the one with the highest variance
            a_i = np.argmax(var_ij)
            # print(f'Question {a_i}')
            if real_s_i[a_i] == 1:
                # update s
                next_s_i = s_i.copy()
                next_s_i[a_i] = 1
            else:
                next_s_i = s_i.copy()
                next_s_i[a_i] = 0
            if j == T_questions-1:
                # compute error
                s_predict = next_s_i.copy()
                s_predict = imputer.transform(np.array([s_predict]))[0]
                # round
                s_predict = np.round(s_predict)
                error = -np.sum(np.abs(s_predict - real_s_i))/(N_questions-T_questions)
                error_greedy_list.append(error)
            s_i = next_s_i
    # compute mean error for all people
    error_greedy = np.mean(error_greedy_list)
    return error_greedy


### add play and record function

def run_experiment(params):
    # Unpack parameters and run the RL simulation
    error = run_RL(**params)
    # Return a dictionary with the results
    result_dict = params.copy()
    result_dict['error'] = error
    return result_dict

def run_RL(k_neighbors = 8, lr = 3e-1, batch_size = 32, start_epsilon = 0.2):

    # read data/df_turkey.csv
    # df_turkey = pandas.read_csv('data/df_turkey.csv', index_col=0, sep=';')
    df_complete = pandas.read_csv('data/data_complete.csv', index_col=0, sep=',')

    # parameters
    N_questions = 15
    T_questions = 6
    train_size = 0.5
    # k_neighbors = 8
    reward_every_question = False
    # lr = 5e-4
    batch_size = 32

    # only take the first N columns
    df_complete = df_complete.iloc[:, :N_questions]


    # separate into train and test randomly
    train, test = train_test_split(df_complete, test_size=1-train_size, random_state=32)
    number_of_people = train.shape[0]

    # train a KNNImputer on train
    imputer = KNNImputer(n_neighbors=k_neighbors)
    imputer.fit(train.to_numpy())


    # test error using a greedy 0 policy
    error_greedy = greedy_0(imputer, train, T_questions)


    verbose = False
    verbose2 = False

    # state_shape = (n_questions,)
    state_shape = (2*N_questions,)

    # define agent
    agent = DQNAgent(state_shape, N_questions, epsilon=0.5).to('cpu')
    target_network = DQNAgent(state_shape, N_questions, epsilon=0.5).to('cpu')
    target_network.load_state_dict(agent.state_dict())

    #setup some parameters for training
    timesteps_per_epoch = 10
    # batch_size = 32
    total_steps = 5 * 10**4

    # number of episodes
    episodes = 4000


    #init Optimizer
    opt = torch.optim.Adam(agent.parameters(), lr=lr)

    # set exploration epsilon 
    # start_epsilon = 0.4
    end_epsilon = 0.01
    eps_decay_final_step = int(episodes/2)

    # setup spme frequency for loggind and updating target network
    # I'm not super sure what these are for
    loss_freq = 20
    refresh_target_network_freq = 50
    initial_buffer_size = 1000
    max_buffer_size = 1100
    # eval_freq = 1000

    # smoothing window for the reward curve
    smoothing_window = 30

    # to clip the gradients
    max_grad_norm = 5000

    # history
    mean_rw_history = []
    td_loss_history = []
    error_list = []


    s_batch = []
    a_batch = []
    r_batch = []
    next_s_batch = []
    done_batch = []
    for _ in range(initial_buffer_size):
        real_s = train.sample().values[0]
        s = np.nan*np.ones(N_questions)
        answered = np.zeros(N_questions)
        state = np.concatenate([imputer.transform(np.array([s]))[0], answered])
        for i in range(T_questions):
            qvalues = agent.get_qvalues([state])
            a = agent.sample_actions(qvalues)[0]
            if real_s[a] == 1:
                next_s = s.copy()
                next_s[a] = 1
            else:
                next_s = s.copy()
                next_s[a] = 0
            next_answered = answered.copy()
            next_answered[a] = 1
            s_predict = next_s.copy()
            s_predict = imputer.transform(np.array([s_predict]))[0]
            s_predict = np.round(s_predict)
            if reward_every_question and i < T_questions - 1:
                r = (0.99**(N_questions-i))*-np.sum(np.abs(s_predict - real_s))/(N_questions-i+1)
            else: 
                r = 0
            if i == T_questions-1:
                r = -np.sum(np.abs(s_predict - real_s))/(N_questions-T_questions)
                done = True
            else:
                done = False
            next_state = np.concatenate([s_predict, next_answered])
            s_batch.append(state)
            a_batch.append(a)
            r_batch.append(r)
            next_s_batch.append(next_state)
            done_batch.append(done)
            state = next_state
            s = next_s
            answered = next_answered




    # run episodes
    for ep in range(episodes):
        if verbose: print(f'Episode {ep}')
        # reduce exploration as we progress
        agent.epsilon = epsilon_schedule(start_epsilon, end_epsilon, ep, eps_decay_final_step)

        # define lists to store batch
        s_list = []
        a_list = []
        r_list = []
        next_s_list = []
        done_list = []

        for batch in range(timesteps_per_epoch):

            # initial s is each question is np.nan
            s = np.nan*np.ones(N_questions)

            # initial s_p
            s_p = imputer.transform(np.array([s]))[0]
            # initial answered
            answered = np.zeros(N_questions)

            # state is a vector that joins them 3
            state = np.concatenate([s_p, answered]) # this probably doesnt work, could just be s_p instead

            # maybe state can be current prediction
            # s_p = imputer.transform(np.array([s]))[0]

            # real_s is one random person from training
            real_s = train.sample().values[0]

            # done flag
            done = False
            # s_list = []
            # r_list = []
            # a_list = []
            # next_s_list = []
            # done_list = []

            # iterate over questions to ask
            for i in range(T_questions):
                # print current state
                if verbose: print('-'*10)
                # if verbose: print(f'Question {i}')
                # if verbose: print(f'State {state}')
                
                # select action
                qvalues = agent.get_qvalues([state])
                a = agent.sample_actions(qvalues)[0]

                # update s with the answer
                if real_s[a] == 1:
                    # update s
                    next_s = s.copy()
                    next_s[a] = 1
                else:
                    next_s = s.copy()
                    next_s[a] = 0

                next_answered = answered.copy()
                next_answered[a] = 1

                # create copy of s for prediction
                s_predict = next_s.copy()
                # # change 0 for nan
                # s_predict[s_predict == 0] = np.nan
                # # change -1 for 0
                # s_predict[s_predict == -1] = 0
                # predict using imputer
                s_predict = imputer.transform(np.array([s_predict]))[0]
                next_s_p = s_predict.copy()

                # if verbose: print(f'Predict probs {s_predict}')
                # round prediction
                s_predict = np.round(s_predict)

                # compute rewarding depending on reward_every_question
                if reward_every_question and i < T_questions - 1:
                    r = (0.99**(N_questions-i))*-np.sum(np.abs(s_predict - real_s))/(N_questions-i+1)
                else: 
                    r = 0
                # if last question compute reward
                if i == T_questions-1:
                    if verbose: print(f'Predict vals {s_predict}')
                    if verbose: print(f'Real s {real_s}')
                    r = -np.sum(np.abs(s_predict - real_s))/(N_questions-T_questions)
                    if verbose: print(f'Reward {r}')
                    done = True

                # create next state
                next_state = np.concatenate([next_s_p, next_answered])
                
                # append to lists
                s_list.append(state)
                a_list.append(a)
                r_list.append(r)
                next_s_list.append(next_state)
                done_list.append(done)

                # update state
                state = next_state
                # update state
                s = next_s

        # extend buffers
        s_batch.extend(s_list)
        a_batch.extend(a_list)
        r_batch.extend(r_list)
        next_s_batch.extend(next_s_list)
        done_batch.extend(done_list)
        # clip to max
        if len(s_batch) > max_buffer_size:
            s_batch = s_batch[-max_buffer_size:]
            a_batch = a_batch[-max_buffer_size:]
            r_batch = r_batch[-max_buffer_size:]
            next_s_batch = next_s_batch[-max_buffer_size:]
            done_batch = done_batch[-max_buffer_size:]

        # all to array
        # s_list = np.array(s_list)
        # a_list = np.array(a_list)
        # r_list = np.array(r_list)
        # next_s_list = np.array(next_s_list)
        # done_list = np.array(done_list)

        # sample from batch
        indices = np.random.choice(len(s_batch), batch_size)
        s_array = np.array(s_batch)[indices]
        a_array = np.array(a_batch)[indices]
        r_array = np.array(r_batch)[indices]
        next_s_array = np.array(next_s_batch)[indices]
        done_array = np.array(done_batch)[indices]


        loss = compute_td_loss(agent, target_network, 
                           s_array, a_array, r_array, next_s_array, done_array,                  
                           gamma=0.99,
                           device='cpu')
        
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
        opt.step()
        opt.zero_grad()


        ### compute error for all people (this should probably be a separate function)
        if ep % refresh_target_network_freq == 0:
            # iterate train
            error = 0
            for person_i in range(train.shape[0]):
                if verbose2: print(f'Persona {person_i}')
                real_s_i = train.iloc[person_i].values
                s_i = np.nan*np.ones(N_questions)
                answered_i = np.zeros(N_questions)
                state_i = np.concatenate([imputer.transform(np.array([s_i]))[0], answered_i])
                for j in range(T_questions):
                    qvalues_i = agent.get_qvalues([state_i])
                    # choose best arg for qvalues
                    a_i = qvalues_i.argmax()
                    if real_s_i[a_i] == 1:
                        # update s
                        next_s_i = s_i.copy()
                        next_s_i[a_i] = 1
                    else:
                        next_s_i = s_i.copy()
                        next_s_i[a_i] = 0

                    next_answered_i = answered_i.copy()
                    next_answered_i[a_i] = 1

                    s_predict_i = next_s_i.copy()
                    s_predict_i = imputer.transform(np.array([s_predict_i]))[0]
                    # round
                    s_predict_i = np.round(s_predict_i)
                    if j == T_questions-1:
                        # persona i print

                       
                        if verbose2: print(f'Predict vals {s_predict_i}')
                        if verbose2: print(f'Real s {real_s_i}')
                        r_i = -np.sum(np.abs(s_predict_i - real_s_i))/(N_questions-T_questions)
                        if verbose2: print(f'Reward {r_i}')
                        if verbose2: print('-'*10)
                    else:
                        r_i = 0

                    if j == T_questions-1:
                        done_i = True
                    else:
                        done_i = False

                    next_state_i = np.concatenate([s_predict_i, next_answered_i])

                    state_i = next_state_i
                    s_i = next_s_i
                    answered_i = next_answered_i
                    error += r_i
            error_list.append(error/train.shape[0])
            # plot error_list
            plt.figure()
            plt.plot(error_list, label = 'RL')
            # add greedy error to plot
            plt.axhline(y=error_greedy, color='r', linestyle='--', label='Greedy 0')
    
            # Adding an annotation with the parameters
            param_text = (f'Parameters:\n'
                        f'LR: {lr}\n'
                        f'Batch Size: {batch_size}\n'
                        f'K-Neighb: {k_neighbors}\n'
                        f'Start Eps: {start_epsilon}\n'
                        f'N People: {number_of_people}\n')

            # Place the text box in upper left in axes coords
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            plt.gca().text(0.05, 0.95, param_text, transform=plt.gca().transAxes, fontsize=10,
                        verticalalignment='top', bbox=props)
            
            # Finish up setting the plot
            plt.legend()
            plt.title('Error until episode ' + str(ep))
            plt.xlabel(f'Episode (once every {refresh_target_network_freq})')
            plt.ylabel('Mean reward')

            # name file with parameters
            plt.savefig(f'output/error_list_E{episodes}_M{T_questions}_N{N_questions}_ts{train_size}_lr{lr}_batch_size{batch_size}_epsilon{start_epsilon}_kneigh{k_neighbors}.png')
            plt.close()

        # 
        if ep % loss_freq == 0:
            mean_rw_history.append(np.mean(r_list))
            td_loss_history.append(loss.data.cpu().item())

        if ep % refresh_target_network_freq == 0:
            # Load agent weights into target_network
            target_network.load_state_dict(agent.state_dict())

            if len(mean_rw_history) > smoothing_window*3:
                # smooth
                mean_rw_history_partial = savgol_filter(mean_rw_history, smoothing_window, 1)

                # save as csv
                # np.savetxt('output/mean_rw_history.csv', mean_rw_history, delimiter=',')
                
                plt.figure()
                plt.plot(mean_rw_history_partial)
                plt.title('Mean reward until episode ' + str(ep))
                # name file with parameters
                plt.savefig(f'output/mean_rw_history_E{episodes}_M{T_questions}_N{N_questions}_ts{train_size}.png')
                plt.close()
                # next plot        
    return error_list[-1]


def main():

    parallel = True

    # parameter grid
    # param_grid = {'k_neighbors': [8, 10, 12], 
    #               'lr': [1e-1, 3e-1, 5e-1], 
    #               'batch_size': [32, 64], 
    #               'start_epsilon': [0.1, 0.4, 0.6]}
    param_grid = {'k_neighbors': [7], 
                  'lr': [1e-3, 1e-4,5e-5],
                  'batch_size': [32, 64],
                  'start_epsilon': [0.4,0.6]}


    # for storing results
    results = []

    # Create a list of all parameter names and their corresponding lists of values
    param_keys, param_values = zip(*param_grid.items())

    # Use itertools.product to generate all combinations of parameter param_values
    combinations = [dict(zip(param_keys, v)) for v in itertools.product(*param_values)]

    if parallel:
        with Pool() as pool:
            results = pool.map(run_experiment, combinations)



    else:
        # iterate over parameter grid
        for params in combinations:
            print(params)
            # run RL
            error = run_RL(**params)
            
            # save results in dictionary to append to results list
            results_dict = params.copy()
            results_dict['error'] = error

            results.append(results_dict)
    
    # save results
    results_df = pandas.DataFrame(results)

    results_df.to_csv('output/results.csv')


        
        # save error
        # save error with parameters

    # iterate over every parameter grid combination with only one for loop


    # iterate over parameter grid


# def main_old():
#     # env = FrameStack(ResizeObservation(GrayScaleObservation(SkipFrame(env, n_frames=4)), shape=84), num_stack=4)

#     # gamma = 0.99 
#     # learning_rate = 0.0001
#     # max_steps = 3
#     # n_repetitions = 2
#     # n_episodes = 20000
#     # smoothing_window = 3
#     # r = run_repetitions('MonteCarloG', n_repetitions, n_episodes, smoothing_window, learning_rate, gamma, max_steps)
#     # # r, error = run_repetitions_v2('MonteCarloG_statevalue', n_repetitions, n_episodes, smoothing_window, learning_rate, gamma, max_steps)
#     # print('finished')
#     # # plot r with matplotlib
#     # print(r)
#     # import matplotlib.pyplot as plt
#     # plt.plot(r)
#     # # add title and axis
#     # plt.title('Learning Curve')
#     # plt.xlabel('Episode')
#     # plt.ylabel('Reward')
#     # # save plot
#     # plt.savefig('Graph_Test.png')


#     exit()
#     # start new plot
#     plt.figure()
#     # plot error
#     print(error)
#     plt.plot(error)
#     # add title and axis
#     plt.title('Final Error')
#     plt.xlabel('Episode')
#     plt.ylabel('Error')
#     # save plot
#     plt.savefig('Graph_Test_error.png')

#     # Plot.add_curve(r, label='')
#     # Plot.save('Graph_Test')


if __name__ == '__main__':
    # set random seed
    np.random.seed(32)
    main()
