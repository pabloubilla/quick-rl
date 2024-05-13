#%%
import numpy as np

# Parameters
S = [10*(i+1) for i in range(5)] # total number of questions
N = [100,400,1000] # number of individuals (answerers)
rho = 0.99 # Pearson correlation coefficient
p = 0.5 # Bernoulli parameter

# Generate synthetic data (Bernoulli pairs):
# Half of S are Bernoulli(0.5) and the other half are correlated with the first half,
# but with a correlation coefficient of rho.
def generate_synthetic_data(S, N, rho, p):
    # Generate the first half of the questions
    X = np.random.binomial(1, p, size=(N, S//2))
    # Generate the second half of the questions: each pair is associated with only one of the 
    # first half questions, with a correlation coefficient of rho
    Y1 = np.zeros((N, S//2))
    Y2 = np.zeros((N, S//2))
    Y3 = np.zeros((N, S//2))
    y1_list = []
    y2_list = []
    y3_list = []
    for i in range(S//2):
        Y1[:, i] = np.random.binomial(1, rho*X[:, i] + (1-rho)*p, N)
        # print(X[:, i], Y1[:, i])
        # print("-"*10 + f"Question {i+1}" + "-"*10)
        y1_list.append(np.corrcoef(X[:, i], Y1[:, i])[0,1])
        # print(np.corrcoef(X[:, i], Y1[:, i])[0,1])

        # Other method: With probability rho, Y is equal to X, and with probability 1-rho
        # change the value to the opposite for each entry of Y[:,i]
        # print('1s in X:', np.mean(X[:, i]))
        # print('p:', p)
        for j in range(N):

            if j <= N*rho:
            # if np.random.uniform() < rho:
                Y2[j, i] = X[j, i]
            # else:
            elif j <= N*rho + N*(1-rho)*p*0.99:
                Y2[j, i] = X[j, i]
                # Y2[j, i] = np.random.binomial(1, p)
            else:
                Y2[j, i] = 1- X[j, i]

        # print(np.corrcoef(X[:, i], Y2[:, i])[0,1])
        y2_list.append(np.corrcoef(X[:, i], Y2[:, i])[0,1])

        # Method 3:
        for j in range(N):
            if j <= 10:
                Y3[j, i] = X[j, i]
            else:
                partial_rho = np.corrcoef(X[:j, i], Y3[:j, i])[0,1]
                # print('partial_rho:', partial_rho)
                if partial_rho > rho:
                    Y3[j, i] = 1 - X[j, i]
                else:
                    Y3[j, i] = X[j, i]
        y3_list.append(np.corrcoef(X[:, i], Y3[:, i])[0,1])
    print("-"*50)
    # print('y1')
    # print(np.mean(y1_list))
    # print(np.std(y1_list))
    print('y2')
    print(np.mean(y2_list))
    print(np.std(y2_list))
    # print('y3')
    # print(np.mean(y3_list))
    # print(np.std(y3_list))

    # Select method
    Y = Y2 

    for j in range(S//2):
        # shuffle paired columns to avoid patterns
        j_indices = np.random.permutation(N)
        X[:, j] = X[j_indices, j]
        Y[:, j] = Y[j_indices, j]

    Z = np.concatenate((X, Y), axis=1)

    print("Correlation matrix of output:")
    print(np.corrcoef(Z.T))

    return Z

for s in S:
    for n in N:
        data = generate_synthetic_data(s, n, rho, p)
        np.savetxt(f'synthetic_data/synthetic_data_{s}_{n}_corr{rho}.csv', data, delimiter=',')
# %%

# Previous names did not have the corr. (rho=0.99)
