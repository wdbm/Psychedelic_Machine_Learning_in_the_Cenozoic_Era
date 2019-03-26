import logging
import re

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from collections.abc import MutableSequence
logging.basicConfig(level=logging.INFO)

class Loggable:
    @property
    def log(self):
        """Property to return a mixin logger."""
        my_log = logging.getLogger('classlog')
        my_log.name = self.__class__.__name__
        return my_log

class SequenceFrame(Loggable):
    """ provides a set of methods which return modified copies of an internal pandas dataframe
        suitable for setting up LSTM problems in keras

        accepts a pd.DataFrame with rows assumed to be ordered by time (uniform steps)

        use:

        (reflection_steps) sequential past observations of (reflection_vars)
        supplemented with knowledge of (lookahead_vars) at each future timestep

        as inputs to predict:

        the following (forecast_steps) of (forecast_vars)
    """
    def __init__(self, df,
                 forecast_vars='.*',
                 reflection_vars='.*',
                 lookahead_vars=[],
                 forecast_steps=1,
                 reflection_steps=1,
                 lookahead_steps=0,
                 step_window=1,
                 time_suffix='t'):
        self.df = df
        self.forecast_vars = forecast_vars
        self.reflection_vars = reflection_vars
        self.lookahead_vars  = lookahead_vars
        self.time_suffix = time_suffix
        self.configure_sequence_steps(forecast_steps,
                                      reflection_steps,
                                      lookahead_steps)
        self.step_window = 1

    @property
    def forecast_vars(self):
        return self._forecast_vars

    @property
    def reflection_vars(self):
        return self._reflection_vars

    @property
    def lookahead_vars(self):
        """ and those future predictions which are "sure things" and are considered inputs
            at each time step (ie these will be given in each sequence alongside the
            sequential past data values. for example, weather 2 days in the future might
            be known to high accuracy alongside past observations of the observable of interest)
        """
        return self._lookahead_vars

    @property
    def forecast_steps(self):
        return self._forecast_steps

    @property
    def reflection_steps(self):
        return self._reflection_steps

    @property
    def max_sequence_length(self):
        """ sequence length is the number of timesteps spanned by any
            of the variables used in training

            ie the prev 10 timesteps of some variable(s) are used to
            predict some (other) variable(s) up to 4 steps in the future,
            this is 14
        """
        return self.reflection_steps + self.forecast_steps - 1

    @forecast_vars.setter
    def forecast_vars(self, var_spec):
        vs = self.matching_columns(var_spec)
        self.log.debug("Setting forecasted variables to:")
        self.log.debug(vs, "\n")
        self._forecast_vars = vs

    @reflection_vars.setter
    def reflection_vars(self, var_spec):
        vs = self.matching_columns(var_spec)
        self.log.debug("Setting reflected variables to:")
        self.log.debug(vs, "\n")
        self._reflection_vars = vs

    @lookahead_vars.setter
    def lookahead_vars(self, var_spec):
        vs = self.matching_columns(var_spec)
        self.log.debug("Setting lookahead variables to:")
        self.log.debug(vs, "\n")
        if not (frozenset(self.forecast_vars).isdisjoint(var_spec)):
            raise ValueError("lookahead vars can't overlap with forecasted vars")
        self._lookahead_vars = vs

    @property
    def reflected_columns(self):
        return [self.var_at_timestep(v, -t) for v in self.reflection_vars
                for t in range(1, self.reflection_steps+1)]

    @property
    def forecasted_columns(self):
        return [self.var_at_timestep(v, t) for v in self.forecast_vars
                for t in range(self.forecast_steps)]

    @property
    def lookahead_columns(self):
        return [self.var_at_timestep(v, t) for v in self.lookahead_vars
                for t in range(self.lookahead_steps)]

    @property
    def lookahead_steps(self):
        return self._lookahead_steps

    def matching_columns(self, var_spec):
        if isinstance(var_spec, str):
            var_spec = [var_spec]
        return [c for c in self.df.columns if any(re.match(re.compile(v), c) for v in var_spec)]

    def configure_sequence_steps(self, forecast_steps=1, reflection_steps=1, lookahead_steps=0):
        try:
            assert forecast_steps + reflection_steps <= len(self.df), (
                "Number of time steps back and forward cannot be more than "
                "the number of consecutive training examples."
            )
            assert lookahead_steps <= forecast_steps, "acausal setup"
        except AssertionError as a:
            self.log.exception(a)
            forecast_steps = 0
            reflection_steps = 0
        self._forecast_steps = forecast_steps
        self._reflection_steps = reflection_steps
        self._lookahead_steps = lookahead_steps

    def timestep_suffix(self, timestep):
        if timestep == 0:
            return f'{self.time_suffix}'
        elif timestep < 0: # - sign already explicit
            return f'{self.time_suffix}{timestep}'
        elif timestep > 0: # add explicit + sign
            return f'{self.time_suffix}+{timestep}'

    def var_at_timestep(self, var, timestep):
        if var not in self.df.columns:
            raise ValueError(f"{var} not found in dataframe!")
        return f'{var}_{self.timestep_suffix(timestep)}'

    def matching_vars_at_timestep(self, var_pat, timestep):
        """ return df w/ all the columns matching var_pat at a given timestep """
        if not hasattr(self, 'shifted_df'):
            self.frame_shift()
        patstr = f'{var_pat}_{self.timestep_suffix(timestep)}$'
        #print('patstr is', patstr)
        pat = re.compile(patstr)
        cols = [c for c in self.shifted_df.columns
                if re.match(re.compile(pat), c)]
        #print("cols are", cols)
        return self.shifted_df[cols]

    def frame_shift(self, dropnan=True):
        """
        Frame a time series as a supervised learning dataset.
        Arguments:
            dropnan: Boolean whether or not to drop rows with NaN values.
        Returns:
            Pandas DataFrame of series framed for supervised learning.
        """
        n_vars = self.df.shape[1]
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in (-x for x in range(self.reflection_steps, 0, -1)):
            cols.append(self.df.shift(i))
            names += [self.var_at_timestep(v, i) for v in self.df.columns]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, self.forecast_steps):
            cols.append(self.df.shift(-i))
            names += [self.var_at_timestep(v, i) for v in self.df.columns]
        # put it all together
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if True:#dropnan:
            agg.dropna(inplace=True)
        self.shifted_df = agg
        return agg

    @property
    def step_window(self):
        """ the size of the steps between each sequence, fixing the amount
            the 'sliding window' moves which defines each training sequence
            in splitting up the data

            for example for 3-length sequences and a step window of 1, the following
            sequence of training data:

            [a,b,c,d,e,f] -> [ [a,b,c], [b,c,d], [c,d,e], [d,e,f]]

            whereas with a window of 3:

            [a,b,c,d,e] -> [[a,b,c], [d,e,f]
        """
        return self._step_window

    @step_window.setter
    def step_window(self, value):
        if value > self.reflection_steps:
            raise ValueError(f"step window of length {value} will miss data "
                             "as this is longer than the sequence length!")
        self._step_window = value

    def reshaped_LSTM_values(self):
        """ convert 2D dataframe to 3D numpy array of requested dimensions
            samples * refl_sequence_length * variables


        """
        # first, get the rows which will be converted into each sample
        # with a given sequence length based on the step window. step window of
        # 1 implies maximum training data usage.
        _df = self.frame_shift(self.df)
        n_samples_total = len(_df)
        row_inds_to_use = [i for i in range(0, n_samples_total, self.step_window)]
        n_samples = len(row_inds_to_use)
        _df = _df.iloc[row_inds_to_use]
        # from each row(<=>sample), extract the sequences of variables to be forecast
        # these are the outputs
        forecasted_variables = _df[[c for c in _df.columns if c in self.forecasted_columns]]
        # and lookahead values which will be included as distinct inputs in each past
        # sequence entry
        lookahead_variables = _df[[c for c in _df.columns if c in self.lookahead_columns]]
        # finally the reflected sequences
        reflected_sequences = _df[[c for c in _df.columns if c in self.reflected_columns]]

        # split the reflected sequences according to the number of timesteps in each
        # (i.e. there should be n_vars * n_steps columns, which we split into a 2d-array)
        # of the shape (n_steps, n_vars). these were appended in order of timesteps
        # so we can just go ahead and reshape...
        input_vals = reflected_sequences.values.reshape(
            (n_samples, self.reflection_steps , len(self.reflection_vars))
            )
        # now splice in the lookahead variables
        # TODO
        # input_vals = ...
        output_vals = forecasted_variables.values
        return input_vals, output_vals

def plot_history(history):

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')

    for s, h in history.items():
        hist = pd.DataFrame(h.history)
        hist['epoch'] = h.epoch

        plt.plot(hist['epoch'], hist['mean_absolute_error'],
                 label=f'Train Error: {s}')
        plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
                 label = f'Val Error: {s}')
        plt.ylim([.5,1.])
        plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')

    for s, h in history.items():
        hist = pd.DataFrame(h.history)
        hist['epoch'] = h.epoch

        plt.plot(hist['epoch'], hist['mean_squared_error'],
             label=f'Train Error: {s}')
        plt.plot(hist['epoch'], hist['val_mean_squared_error'],
             label = f'Val Error: {s}')
        plt.legend()

        plt.ylim([1,3])
        plt.legend()
    plt.show()
