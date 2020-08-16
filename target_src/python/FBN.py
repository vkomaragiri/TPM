import numpy as np
#from sklearn.linear_model import LogisticRegressionCV
import sys, os, pickle
from keras.layers import Input, Dense
from keras import Model
from keras.callbacks import EarlyStopping

class Variable:
    def __init__(self, id_=-1, d_=-1, val_=-1, is_evid_=False, t_val_=-1):
        self.id = id_
        self.d = d_
        self.is_evid = is_evid_
        self.val = val_
        self.t_val = t_val_

class Function:
    def __init__(self, variables_ = [], potentials_ = [], cpt_var_=-1, type_=0): #type=0: CPT
        self.cpt_var = cpt_var_
        self.variables = variables_
        self.potentials = potentials_
        self.type = type_

    def getPotential(self, d):
        if self.type == 2:
            #res = np.array([self.potentials[x[self.cpt_var.id]] for x in d], dtype = np.float)
            res = np.zeros((d.shape[0]))
            ind = d[:, self.cpt_var.id]
            for val in range(self.cpt_var.d):
                res[ind == val] = self.potentials[val]
        else:
            indices = []
            dsize = []
            for var in self.variables:
                indices.append(var.id)
                dsize.append(var.d)
            x = d[:, indices]
            if self.type == 0:
                ind = getAddr(x, dsize)
                #res = np.array([self.potentials[i] for i in ind])
                res = np.zeros((d.shape[0]))
                vals = np.unique(ind)
                for val in vals:
                    res[ind == val] = self.potentials[val]
            elif self.type == 1:
                #res = self.potentials.predict_proba(x[:, :-1])[np.arange(x.shape[0]), x[:, -1]]
                res = self.potentials.predict(x[:, :-1])[np.arange(x.shape[0]), x[:, -1]]

        return res


    def getLogPotential(self, d):
        return np.log(self.getPotential(d))

class FBN:
    def __init__(self):
        self.variables = []
        self.functions = []
        self.order = []
        self.nvariables = -1
        self.dsize = []

    def learn(self, data, valid_data):
        self.nvariables = data.shape[1]
        self.dsize = np.array(np.amax(data, axis=0), dtype=np.int64)+np.ones(self.nvariables, dtype=np.int64)
        print(self.dsize)
        for i in range(self.nvariables):
            self.variables.append(Variable(i, self.dsize[i]))
        self.order = list(range(self.nvariables))
        np.random.shuffle(self.order)
        for i in range(self.nvariables):
            func = Function(variables_=list(np.array(self.variables)[self.order[:i+1]]), cpt_var_=self.variables[self.order[i]])
            if i < 6:
                self.getCPT(i, data, func)
            else:
                self.getNN(i, data, func, valid_data)
            self.functions.append(func)

    def print_BN(self):
        print(self.nvariables)
        print("order: ", self.order, " ", "domains: ", self.dsize)
        print("variables: ", self.variables)
        print("cpds: ", self.functions)

    def getCPT(self, i, data, func):
        num = np.bincount(np.apply_along_axis(getAddr, 1, data[:, self.order[:i+1]], self.dsize[self.order[:i+1]]), minlength=np.prod(self.dsize[self.order[:i+1]]))
        #num = np.bincount(np.apply_along_axis(getAddr, 1, func.variables))
        num += np.ones(num.shape, dtype=np.int64)
        den = np.bincount(np.apply_along_axis(getAddr, 1, data[:, self.order[:i]], self.dsize[self.order[:i]]), minlength=np.prod(self.dsize[self.order[:i]]))
        den += self.dsize[self.order[i]]*np.ones(den.shape, dtype=np.int64)
        den = np.array(np.repeat(den, self.dsize[self.order[i]]), dtype = np.float128)
        func.potentials = np.divide(num, den)
        func.type = 0
    '''
    def getLR(self, i, data, func, valid_data):
        X = data[:, self.order[:i]]
        y = data[:, self.order[i]]
        print(X.shape, y.shape)
        if (len(np.unique(y))) > 1:
            #print(y, y.shape[0], X.shape[1])
            clf = LogisticRegressionCV(cv=5, Cs=10, fit_intercept=True, penalty='elasticnet', solver='saga', tol=0.0001, max_iter=1000, n_jobs=8, verbose=0, multi_class='multinomial', l1_ratios=[0, 0.2, 0.4, 0.6, 0.8, 1]).fit(X, y)
            X_valid = valid_data[:, self.order[:i]]
            y_valid = valid_data[:, self.order[i]]
            print(clf.score(X_valid, y_valid))
            #func.potentials = np.hstack([np.reshape(clf.intercept_, (clf.intercept_.shape[0], 1)), clf.coef_])
            func.potentials = clf
            func.type = 1
        else:
            func.potentials = np.zeros(self.dsize[self.order[i]])
            for val in range(self.dsize[self.order[i]]):
                if val == y[0]:
                    func.potentials[y[0]] = 0.99
                else:
                    func.potentials[val] = 0.01/(self.dsize[self.order[i]]-1)
            func.type = 2
    '''
    def getNN(self, i, data, func, valid_data):
        X = data[:, self.order[:i]]
        y = data[:, self.order[i]]
        print(X.shape, y.shape)
        if (len(np.unique(y))) > 1:
            X_valid = valid_data[:, self.order[:i]]
            y_valid = valid_data[:, self.order[i]]
            #print(y, y.shape[0], X.shape[1])
            #clf = LogisticRegressionCV(cv=5, Cs=10, fit_intercept=True, penalty='elasticnet', solver='saga', tol=0.0001, max_iter=1000, n_jobs=8, verbose=0, multi_class='multinomial', l1_ratios=[0, 0.2, 0.4, 0.6, 0.8, 1]).fit(X, y)
            input = Input(shape=(X.shape[1],))
            dense1 = Dense(int(X.shape[1]//2+1), activation='relu') (input)
            #dense2 = keras.layers.Dense(int(X.shape[1]//3+1), activation='relu') (dense1)
            output = Dense(len(np.unique(y)), activation='softmax') (dense1)
            model = Model(inputs=input, outputs=output)
            model.summary()
            callback = [EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')]
            model.compile(optimizer="sgd", metrics = ["accuracy"], loss="sparse_categorical_crossentropy")
            model.fit(x=X, y=y, epochs=100, verbose=2, validation_data=(X_valid, y_valid), shuffle=True, callbacks=callback)
            #func.potentials = np.hstack([np.reshape(clf.intercept_, (clf.intercept_.shape[0], 1)), clf.coef_])
            func.potentials = model
            func.type = 1
        else:
            func.potentials = np.zeros(self.dsize[self.order[i]])
            for val in range(self.dsize[self.order[i]]):
                if val == y[0]:
                    func.potentials[y[0]] = 0.99
                else:
                    func.potentials[val] = 0.01/(self.dsize[self.order[i]]-1)
            func.type = 2

    def getProbability(self, d):
        return np.prod(np.array([f.getPotential(d) for f in self.functions]), axis = 0)

    def likelihood(self, data):
        l = np.prod(self.getProbability(data))/data.shape[0]
        return l

    def getLogProbability(self, d):
        return np.sum(np.array([f.getLogPotential(d) for f in self.functions]), axis = 0)

    def log_likelihood(self, data):
        ll = np.sum(self.getLogProbability(data))/data.shape[0]
        return ll

    def setValue(self, var, val):
        self.variables[var].val = val
        self.variables[var].is_evid = True

    def generateSamples(self, n):
        samples = np.zeros((n, len(self.dsize)), dtype=np.int)
        weights = np.ones(n)
        for j in self.order:
            prob = np.zeros((n,1))
            #print(j, self.order.index(j), self.functions[self.order.index(j)].cpt_var.id)
            for k in range(self.dsize[j]):
                samples[:, j] = k
                pot = self.functions[self.order.index(j)].getPotential(samples)
                pot = np.reshape(pot, (pot.shape[0], 1))
                prob = np.hstack([prob, pot])
            prob = prob[:, 1:]
            if self.variables[j].is_evid:
                samples[:, j] = self.variables[j].val
                weights *= prob[:, self.variables[j].val]
            else:
                random_numbers = np.random.ranf(n)
                for val in range(self.dsize[j]):
                    samples[np.logical_and(random_numbers < np.sum(prob[:, :val+1], axis = 1), random_numbers >= np.sum(prob[:, :val], axis = 1)), j] = val
                    #samples[:, j] =
                #samples[:, self.order[j]] = np.random.choice(self.dsize[self.order[j]], size = 1, p = prob)[0]
        return samples, weights


def readCSVData(fpath):
    return np.loadtxt(fpath, delimiter=',', dtype=np.int64)


def getAddr(arr, dsize):
    if arr.ndim == 2:
        ind = np.zeros(arr.shape[0], dtype=np.int)
        multiplier = np.ones(arr.shape[0], dtype=np.int)
        for i in range(arr.shape[1]-1, -1, -1):
            ind += np.multiply(multiplier, arr[:, i])
            multiplier *= dsize[i]
    else:
        ind = 0
        multiplier = 1
        for i in range(arr.shape[0]-1, -1, -1):
            ind += np.multiply(multiplier, arr[i])
            multiplier *= dsize[i]
    return ind

def main():
    if len(sys.argv) < 6:
        print("Usage:\npython FBN.py <Train Data Path> <Validation Data Path> <Test Data Path> <Model Path> <Test LL File Path>")
        exit(0)
    train_data = readCSVData(sys.argv[1])
    valid_data = readCSVData(sys.argv[2])
    test_data = readCSVData(sys.argv[3])
    bn = FBN()
    bn.learn(train_data, valid_data)

    f = open(sys.argv[5], "a")
    f.write(sys.argv[1].split('.')[0]+" "+str(bn.log_likelihood(test_data))+"\n")
    filehandler = open(sys.argv[4], 'wb')
    pickle.dump(bn, filehandler)
    '''
    bn = pickle.load(open(sys.argv[4], 'rb'))

    data = train_data[:2, :]
    for f in bn.functions:
        print(f.getPotential(data))
    print(bn.generateSamples(2))
    bn.setValue(5, 1)
    bn.setValue(11, 1)
    print(bn.generateSamples(100)[1])

    for var in bn.variables:
        print(var.is_evid, end = " ")
    
    data = train_data[:2, :]
    for f in bn.functions:
        print(f.getPotential(data))
    print(bn.generateSamples(2))
    bn.setValue(5, 1)
    bn.setValue(11, 1)
    print(bn.generateSamples(100)[1])

    for var in bn.variables:
        print(var.is_evid, end = " ")
    '''
if __name__ == '__main__':
    main()





