# Wrappers for the Scikit-Learn API

You can use `Sequential` bidu models (single-input only) as part of your Scikit-Learn workflow via the wrappers found at `bidu.wrappers.scikit_learn.py`.

There are two wrappers available:

`bidu.wrappers.scikit_learn.biduClassifier(build_fn=None, **sk_params)`, which implements the Scikit-Learn classifier interface,

`bidu.wrappers.scikit_learn.biduRegressor(build_fn=None, **sk_params)`, which implements the Scikit-Learn regressor interface.

### Arguments

- __build_fn__: callable function or class instance
- __sk_params__: model parameters & fitting parameters

`build_fn` should construct, compile and return a bidu model, which
will then be used to fit/predict. One of the following
three values could be passed to build_fn:

1. A function
2. An instance of a class that implements the __call__ method
3. None. This means you implement a class that inherits from either
`biduClassifier` or `biduRegressor`. The __call__ method of the
present class will then be treated as the default build_fn.

`sk_params` takes both model parameters and fitting parameters. Legal model
parameters are the arguments of `build_fn`. Note that like all other
estimators in scikit-learn, 'build_fn' should provide default values for
its arguments, so that you could create the estimator without passing any
values to `sk_params`.

`sk_params` could also accept parameters for calling `fit`, `predict`,
`predict_proba`, and `score` methods (e.g., `nb_epoch`, `batch_size`).
fitting (predicting) parameters are selected in the following order:

1. Values passed to the dictionary arguments of
`fit`, `predict`, `predict_proba`, and `score` methods
2. Values passed to `sk_params`
3. The default values of the `bidu.models.Sequential`
`fit`, `predict`, `predict_proba` and `score` methods

When using scikit-learn's `grid_search` API, legal tunable parameters are
those you could pass to `sk_params`, including fitting parameters.
In other words, you could use `grid_search` to search for the best
`batch_size` or `nb_epoch` as well as the model parameters.
