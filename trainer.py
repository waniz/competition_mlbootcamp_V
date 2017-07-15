from hyperopt import fmin, tpe, hp, STATUS_OK


def objective(x):
    return {'loss': x ** 2, 'status': STATUS_OK}

best = fmin(objective,
            space=hp.quniform('x', -10, 10, 0.01),
            algo=tpe.suggest,
            max_evals=100)

print(best)
