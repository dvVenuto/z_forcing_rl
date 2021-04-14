from __future__ import print_function

import tensorflow as tf
import numpy as np
import math

from RIMCell_dist import RIMCell

def log_prob_gaussian(x, mu, log_vars, mean=False):
    lp = - 0.5 * math.log(2 * math.pi) \
         - log_vars / 2 - (x - mu) ** 2 / (2 * tf.exp(log_vars))
    if mean:
        return tf.reduce_mean(lp, -1)
    return tf.reduce_sum(lp, -1)


def log_prob_bernoulli(x, mu):
    lp = x * tf.log(mu + 1e-5) + (1. - y) * tf.log(1. - mu + 1e-5)
    return lp


def gaussian_kld(mu_left, logvar_left, mu_right, logvar_right):
    """
    Compute KL divergence between a bunch of univariate Gaussian distributions
    with the given means and log-variances.
    We do KL(N(mu_left, logvar_left) || N(mu_right, logvar_right)).
    """
    gauss_klds = 0.5 * (logvar_right - logvar_left +
                        (tf.exp(logvar_left) / tf.exp(logvar_right)) +
                        ((mu_left - mu_right) ** 2.0 / tf.exp(logvar_right)) - 1.0)



    assert len(gauss_klds.shape) == 2
    return tf.reduce_sum(gauss_klds, 1)

def tf_index_select(input_, dim, indices):
    """
    input_(tensor): input tensor
    dim(int): dimension
    indices(list): selected indices list
    """
    shape = input_.get_shape().as_list()
    if dim == -1:
        dim = len(shape)-1
    shape[dim] = 1

    tmp = []
    for idx in indices:
        begin = [0]*len(shape)
        begin[dim] = idx
        tmp.append(tf.slice(input_, begin, shape))
    res = tf.concat(tmp, axis=dim)

    return res


class Z_Forcing_RIMs(object):
    def __init__(self, input_dim, embedding_dim, rnn_dim, mlp_dim, z_dim, out_dim, output_type="gaussian",cond_ln=False, num_layers=1,z_force=True,embedding_dropout=0. ):
        self.input_dim = input_dim


        self.embedding_dim = embedding_dim
        self.rnn_dim = rnn_dim
        self.mlp_dim = mlp_dim
        self.z_dim = z_dim
        self.num_layers = num_layers
        self.embedding_dropout = embedding_dropout
        self.cond_ln = False
        self.z_force= z_force
        self.out_type=output_type
        self.use_l2=False

        if output_type == 'bernoulli' or output_type == 'softmax':
            self._emb_mod = tf.keras.Sequential([
                tf.keras.Input(shape=self.input_dim),
                tf.keras.layers.Embedding(self.embedding_dim),
                tf.keras.layers.Dropout(self.embedding_dropout),
            ])
        else:
            self._emb_mod = tf.keras.Sequential([
                tf.keras.Input(shape=self.input_dim),
                tf.keras.layers.Dense(self.embedding_dim),
                tf.keras.layers.Dropout(self.embedding_dropout),
            ])

        self.bwd_mod = tf.keras.layers.LSTM(self.rnn_dim,return_sequences=True)
        if not cond_ln:
            cell_fwd = RIMCell(units=self.rnn_dim, nRIM=6, k=4, num_input_heads=1, input_key_size=32,
                               input_value_size=32, input_query_size=32, input_keep_prob=0.9, num_comm_heads=4,
                               comm_key_size=32, comm_value_size=32, comm_query_size=32, comm_keep_prob=0.9)

            self.fwd_mod = tf.keras.layers.RNN(cell_fwd, return_state=True)


        else:
            cell_fwd = RIMCell(units=self.rnn_dim, nRIM=6, k=4, num_input_heads=1, input_key_size=32,
                                       input_value_size=32, input_query_size=32, input_keep_prob=0.9, num_comm_heads=4,
                                       comm_key_size=32, comm_value_size=32, comm_query_size=32, comm_keep_prob=0.9)

            self.fwd_mod = tf.keras.layers.RNN(cell_fwd,return_state=True)

        self.fwd_out_mod = tf.keras.layers.Dense(units= out_dim)


        self.bwd_out_mod = tf.keras.layers.Dense(units=out_dim)

        self._rim_mod = tf.keras.Sequential([
            tf.keras.Input(shape=(self.z_dim)),
            tf.keras.layers.Dense(self.mlp_dim,activation='relu'),
            tf.keras.layers.Dense(units= 6),
        ])

        self._aux_mod = tf.keras.Sequential([
            tf.keras.Input(shape=(self.z_dim + self.rnn_dim * 6)),
            tf.keras.layers.Dense(self.mlp_dim,activation='relu'),
            tf.keras.layers.Dense(units= self.rnn_dim *2),
        ])

        self._gen_mod = tf.keras.layers.Dense(self.mlp_dim,activation='relu')


        self._inf_mod = tf.keras.Sequential([
            tf.keras.Input(shape=(self.rnn_dim * 7)),
            tf.keras.layers.Dense(self.mlp_dim,activation='relu'),
            tf.keras.layers.Dense(units=z_dim * 2),
        ])

        self._pri_mod = tf.keras.Sequential([
            tf.keras.layers.Dense(self.mlp_dim,activation='relu'),
            tf.keras.layers.Dense(units=self.z_dim * 2),
        ])




    #def _gen_mod(self, inputs, cond_ln=self.cond_ln):
    #    if cond_ln:
    #        temp = self._linear(self.z_dim, self.mlp_dim)
    #        temp = self._LReLU(temp)
    #        temp = self._linear(self.mlp_dim, 8 * self.rnn_dim)
    #        return temp
    #    else:
    #        return self._linear(self.z_dim, self.mlp_dim)

    #def _inf_mod(self, inputs):
    #    temp = self._linear(self.rnn_dim * 2, self.mlp_dim)
    #    temp = self._LReLU(temp)
    #    temp = self._linear(mlp_dim, z_dim * 2)
    #    return temp

    #def _pri_mod(self, inputs):
    #    temp = self._linear(inputs, self.mlp_dim)
    #    temp = self._LReLU(temp)
    #    temp = self._linear(temp, self.z_dim * 2)
    #    return temp





    def _init_matrix(self, shape):
        return tf.random.normal(shape, stddev=0.1)


    def reparametrize(self, mu, logvar, eps=None):
        std = tf.exp(tf.math.scalar_mul(0.5,logvar))
        if eps is None:
            eps = std.data.new(std.size()).normal_()
        return tf.math.add(tf.math.multiply(eps,std),mu)

    def fwd_pass(self, x_fwd, hidden, bwd_states=None, z_step=None):
        with tf.name_scope('forward-pass'):


            self.x_fwd = self._emb_mod(x_fwd)
            n_steps = self.x_fwd.shape[1]

            states = [(hidden[0][0], hidden[1][0])]

            states_h = hidden[0]

            states_c = hidden[1]
            klds, zs, log_pz, log_qz, aux_cs = [], [], [], [], []



            eps = tf.random.normal([n_steps])

            assert (z_step is None) or (n_steps == 1)

            for step in range(n_steps):


                h_step = states_h[:,step]
                c_step =states_c[:,step]


                x_step = self.x_fwd[:,step]


                r_step = eps[step]


                pri_params = self._pri_mod(h_step)
                pri_params = tf.clip_by_value(pri_params, -8., 8.)
                pri_mu, pri_logvar = tf.split(pri_params, 2, axis=1)

                if bwd_states is not None:

                    b_step = bwd_states[:,step]

                    inf_params = self._inf_mod(tf.concat((h_step, b_step), axis=1))
                    inf_params = tf.clip_by_value(inf_params, -8., 8.)
                    inf_mu, inf_logvar = tf.split(inf_params, 2, axis=1)
                    kld = gaussian_kld(inf_mu, inf_logvar, pri_mu, pri_logvar)
                    z_step = self.reparametrize(inf_mu, inf_logvar, eps=r_step)

                    if self.z_force:
                        h_step_ = h_step * 0.
                    else:
                        h_step_ = h_step


                    aux_params = self._aux_mod(tf.concat((h_step_, z_step), axis=1))
                    aux_params = tf.clip_by_value(aux_params, -8., 8.)
                    aux_mu, aux_logvar = tf.split(aux_params, 2, axis=1)

                    b_step_ = tf.stop_gradient(b_step)

                    if self.use_l2:
                        aux_step = tf.reduce_sum((b_step_ - tf.tanh(aux_mu)) ** 2.0, 1)
                    else:

                        aux_step = -log_prob_gaussian(b_step_, tf.tanh(aux_mu), aux_logvar, mean=False)

                else:
                    if z_step is None:
                        z_step = self.reparametrize(pri_mu, pri_logvar, eps=r_step)

                    aux_step = tf.reduce_sum(pri_mu * 0., -1)
                    inf_mu, inf_logvar = pri_mu, pri_logvar
                    kld = aux_step


                i_step = self._gen_mod(z_step)

                RIM_dist = self._rim_mod(z_step)



                if self.cond_ln:
                    i_step = tf.clip_by_value(i_step, -3, 3)
                    gain_hh, bias_hh = tf.split(i_step, 2, axis=1)
                    gain_hh = 1. + gain_hh
                    h_new, _,  c_new = self.fwd_mod(x_step, (h_step, c_step),gain_hh=gain_hh, bias_hh=bias_hh)

                else:

                    c_step = tf.convert_to_tensor(c_step)
                    c_step=tf.dtypes.cast(c_step, tf.float32)

                    h_step = tf.convert_to_tensor(h_step)
                    h_step=tf.dtypes.cast(h_step, tf.float32)

                    rnn_input=tf.concat((i_step, x_step), axis=1)


                    c_step=tf.reshape(c_step,[c_step.shape[0],6,int(c_step.shape[1]/ 6)])
                    h_step=tf.reshape(h_step,[h_step.shape[0],6,int(h_step.shape[1] / 6)])


                    rnn_input=tf.reshape(rnn_input,[rnn_input.shape[0],1,rnn_input.shape[1]])
                    RIM_dist=tf.reshape(RIM_dist,[RIM_dist.shape[0],1,RIM_dist.shape[1]])




                    rnn_input = tf.concat([rnn_input, RIM_dist],axis=2)

                    h_new, hidden_s, c_new = self.fwd_mod(rnn_input,[h_step, c_step])

                    c_new=tf.reshape(c_new,[c_new.shape[0],int(6 * c_new.shape[2])])




                states.append((h_new, c_new))
                klds.append(kld)
                zs.append(z_step)
                aux_cs.append(aux_step)
                log_pz.append(log_prob_gaussian(z_step, pri_mu, pri_logvar))


                log_qz.append(log_prob_gaussian(z_step, inf_mu, inf_logvar))



            states=states[1:]



            klds = tf.stack(klds, 0)
            aux_cs = tf.stack(aux_cs, 0)
            log_pz = tf.stack(log_pz, 0)
            log_qz = tf.stack(log_qz, 0)
            zs = tf.stack(zs, 0)

            outputs = [s[0] for s in states]
            outputs = tf.stack(outputs, 0)
            outputs = tf.reshape(outputs,[outputs.shape[1],outputs.shape[0],outputs.shape[2]])

            klds = tf.reshape(klds,[klds.shape[1],klds.shape[0]])
            aux_cs = tf.reshape(aux_cs,[aux_cs.shape[1],aux_cs.shape[0]])
            log_pz = tf.reshape(log_pz,[log_pz.shape[1],log_pz.shape[0]])
            log_qz = tf.reshape(log_qz,[log_qz.shape[1],log_qz.shape[0]])

            zs = tf.reshape(zs,[zs.shape[1],zs.shape[0],zs.shape[2]])

            #zs = tf.reshape(zs,[zs.shape[1],zs.shape[0]])




            outputs = self.fwd_out_mod(outputs)
            return outputs, states, klds, aux_cs, zs, log_pz, log_qz


    def infer(self, x, hidden):
        x_ = x[:-1]
        y_ = x[1:]
        bwd_states, bwd_outputs = self.bwd_pass(x_, y_, hidden)
        fwd_outputs, fwd_states, klds, aux_nll, zs, log_pz, log_qz = self.fwd_pass(
            x_, hidden, bwd_states=bwd_states)
        return zs


    def bwd_pass(self, x, y, hidden):


        idx = np.arange(np.size(y,1))[::-1].tolist()



        # invert the targets and revert back
        #x_bwd = tf_index_select(tf.convert_to_tensor(y), 0, idx)

        x_bwd=tf.gather(
            tf.convert_to_tensor(y), idx, axis=1
        )


        x= tf.convert_to_tensor(x)



        x_bwd = tf.concat((x_bwd, x[:,:1]), axis=1)


        x_bwd = self._emb_mod(x_bwd)



        states = self.bwd_mod(x_bwd)



        outputs = self.bwd_out_mod(states[:,:-1])

        #states = tf_index_select(states,0, idx)

        states=tf.gather(
            states, idx, axis=1
        )


        outputs =tf.gather(
            outputs, idx, axis=1
        )


        return states, outputs


    def call(self, x, y, x_mask, hidden, return_stats=False):
        nbatch, nsteps = tf.shape(x)[0], tf.shape(x)[1]
        bwd_states, bwd_outputs = self.bwd_pass(x, y, hidden)
        fwd_outputs, fwd_states, klds, aux_nll, zs, log_pz, log_qz = self.fwd_pass(
            x, hidden, bwd_states=bwd_states)
        kld = tf.reduce_sum((klds * x_mask), axis=0)
        log_pz = tf.reduce_sum(log_pz * x_mask, axis=0)
        log_qz = tf.reduce_sum(log_qz * x_mask, axis=0)
        aux_nll = tf.reduce_sum(aux_nll * x_mask, axis=0)

        if self.out_type == 'gaussian':

            out_mu, out_logvar = tf.split(fwd_outputs, 2, axis=-1)

            fwd_nll = -log_prob_gaussian(y, out_mu, out_logvar)
            fwd_nll = tf.reduce_sum(fwd_nll * x_mask, axis=0)
            out_mu, out_logvar = tf.split(bwd_outputs, 2, axis=-1)
            bwd_nll = -log_prob_gaussian(x, out_mu, out_logvar)
            bwd_nll = tf.reduce_sum(bwd_nll * x_mask, axis=0)

        elif self.out_type == 'softmax':
            fwd_out = tf.reshape(fwd_outputs, [nsteps * nbatch, self.out_dim])
            fwd_out = tf.nn.log_softmax(fwd_out)
            y = tf.reshape(y, [-1, 1])
            fwd_nll = tf.squeeze(torch.gather(fwd_out, y, axis=1), axis=1)
            fwd_nll = tf.reshape(fwd_nll, [nsteps, nbatch])
            fwd_nll = tf.reduce_sum(-(fwd_nll * x_mask), axis=0)
            bwd_out = tf.reshape(bwd_outputs, [nsteps * nbatch, self.out_dim])
            bwd_out = tf.nn.log_softmax(bwd_out)
            x = tf.reshape(x, [-1, 1])
            bwd_nll = tf.squeeze(torch.gather(bwd_out, x, axis=1), axis=1)
            bwd_nll = tf.reshape(-bwd_nll, [nsteps, nbatch])
            bwd_nll = tf.reduce_sum((bwd_nll * x_mask), axis=0)

        if return_stats:
            return fwd_nll, bwd_nll, aux_nll, kld, log_pz, log_qz

        return tf.reduce_mean(fwd_nll), tf.reduce_mean(bwd_nll), tf.reduce_mean(aux_nll), tf.reduce_mean(kld)


    def generate_onestep(self, x_fwd, x_mask, hidden):
        nsteps, nbatch = x_fwd.size(0), x_fwd.size(1)
        #bwd_states, bwd_outputs = self.bwd_pass(x_bwd, hidden)
        fwd_outputs, fwd_states, klds, aux_nll, zs, log_pz, log_qz = self.fwd_pass(
                    x_fwd, hidden)

        #CHANGE TO TD
        output_prob = F.softmax(fwd_outputs.squeeze(0))
        sampled_output = torch.multinomial(output_prob, 1)
        hidden = (fwd_states[0][0].unsqueeze(0), fwd_states[0][1].unsqueeze(0))
        if self.return_loss:
            return (sampled_output, hidden, aux_nll)
        else:
            return (sampled_output, hidden)


