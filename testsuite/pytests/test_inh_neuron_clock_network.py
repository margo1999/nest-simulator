# -*- coding: utf-8 -*-
#
# test_clopath_aief_cond_alpha.py
#


"""
Test behavior of inhibitory RNN neuron (iaf_cond_alpha + parameter modifications)
"""

import unittest
import nest
import matplotlib.pyplot as plt
import numpy as np

HAVE_GSL = nest.ll_api.sli_func("statusdict/have_gsl ::")


@nest.ll_api.check_stack
@unittest.skipIf(not HAVE_GSL, 'GSL is not available')
class ClopathNeuronTestCase(unittest.TestCase):
    """Test Inhibitory RNN neuron of Maes network (Maes et al. 2020) """

    def test_AnalyseBehaviourByPlotting(self):
        nest.set_verbosity('M_WARNING')
        nest.ResetKernel()
        nest.resolution = 0.01

        neuron_param = {'C_m': 300.0,               # pF
                    'E_L': -62.0,               # mV
                    'E_ex': 0.0,                # mV
                    'E_in': -75.0,              # mV
                    'g_L': 300.0 / 20.0,        # nS
                    'I_e': 0.0,                 # mV          
                    'V_reset': -60.0,           # mV
                    'V_th': -52.0               # mV
                    }
        nest.CopyModel('iaf_cond_alpha', 'inh_RNN_neuron', params=neuron_param)
        print(nest.GetDefaults('inh_RNN_neuron'))
        nrns = nest.Create('inh_RNN_neuron', 2)

        spike_times = [10.0, 20.0, 30.0, 40.0] # [float(i) for i in range(10, 25)]
        sg = nest.Create('spike_generator', 1, {'spike_times': spike_times})

        # Create all necessary recorders
        mm = nest.Create('multimeter', 2, params={
                         'record_from': ['g_ex', 'g_in', 'V_m'], 'interval': 0.01})
        sr = nest.Create('spike_recorder')
        wr = nest.Create('weight_recorder')

        nest.CopyModel('static_synapse', 'static_synapse_wr',
                       {"weight_recorder": wr, "weight": 1.})

        #nest.CopyModel('vogel_sprekeler_synapse', 'vogel_sprekeler_synapse_wr',
        #               {'weight_recorder': wr, 'weight': 1.})

        # Connect all nodes
        nest.Connect(nrns, nrns, conn_spec={'rule': 'all_to_all', 'allow_autapses': False}, syn_spec='static_synapse_wr')
        nest.Connect(sg, nrns[0], syn_spec={'weight':500.0})
        nest.Connect(mm[0], nrns[0])
        nest.Connect(mm[1], nrns[1])
        nest.Connect(nrns, sr)

        nest.Simulate(1000)

        # get recordings
        events_nrn1 = nest.GetStatus(mm[0])[0]['events']
        time_nrn1 = events_nrn1['times']
        events_nrn2 = nest.GetStatus(mm[1])[0]['events']
        time_nrn2 = events_nrn2['times']

        # TODO make init_weight generalized -> use nest.GetConnections before simulation 
        init_weight = nest.GetDefaults('clopath_synapse', keys='weight')
        nrns_conns = nest.GetConnections(nrns, nrns)
        exc_conns_senders = np.array(nrns_conns.source)
        exc_conns_targets = np.array(nrns_conns.target)
        exc_conns_weights = np.array(nrns_conns.weight)

        # t_log = nest.GetStatus(wr, "events")[0]["times"]
        # w_log = nest.GetStatus(wr, "events")[0]["weights"]

        weight_matrix = np.zeros((2, 2))
        weight_matrix[exc_conns_senders-1, exc_conns_targets-1] = exc_conns_weights
        diff_weight_matrix = weight_matrix - init_weight

        spike_events = sr.events

        # plotting results
        plt.rcParams['axes.grid'] = True
        figure, axis = plt.subplots(2, 3)

        axis[0, 0].plot(time_nrn1, events_nrn1['V_m'], label='neuron id = 1')
        axis[0, 0].plot(time_nrn2, events_nrn2['V_m'], label='neuron id = 2')
        axis[0, 0].set_title('Membrane potential [mV]')
        axis[0, 0].set_xlabel('time [ms]')
        axis[0, 0].set_ylabel('Membrane potential [mV]')
        axis[0, 0].legend()

        axis[0, 1].plot(time_nrn1, events_nrn1['g_ex'], label='g_ex_1')
        axis[0, 1].plot(time_nrn1, events_nrn1['g_in'], label='g_in_1')
        axis[0, 1].plot(time_nrn2, events_nrn2['g_ex'], label='g_ex_2')
        axis[0, 1].plot(time_nrn2, events_nrn2['g_in'], label='g_in_2')
        axis[0, 1].set_title('Synaptic conductance [nS]')
        axis[0, 1].set_xlabel('time [ms]')
        axis[0, 1].set_ylabel('Synaptic conductance [nS]')
        axis[0, 1].legend()

        axis[0, 2].scatter(spike_events['times'], spike_events['senders'], s=5.0)
        axis[0, 2].set_xlim(*axis[0, 0].get_xlim())
        axis[0, 2].set_title('spikes')
        axis[0, 2].set_xlabel('time [ms]')
        axis[0, 2].set_ylabel('neuron id')

        # axis[1, 0].plot(time_nrn1, events_nrn1['w'], label='neuron id = 1')
        # axis[1, 0].plot(time_nrn2, events_nrn2['w'], label='neuron id = 2')
        # axis[1, 0].set_title('adaption current [pA]')
        # axis[1, 0].set_xlabel('time [ms]')
        # axis[1, 0].set_ylabel('adaption current [pA]')
        # axis[1, 0].legend()

        axis[1, 1].imshow(weight_matrix, extent=(0.5, 2.5 , 0.5, 2.5))
        axis[1, 1].set_xticks([1, 2])
        axis[1, 1].set_yticks([1, 2])
        axis[1, 1].set_title('weight matrix')
        axis[1, 1].set_xlabel('target [neuron id]')
        axis[1, 1].set_ylabel('sender [neuron id]')
        axis[1, 1].grid(False)

        axis[1, 2].imshow(diff_weight_matrix, extent=(0.5, 2.5 , 0.5, 2.5))
        axis[1, 2].set_xticks([1, 2])
        axis[1, 2].set_yticks([1, 2])
        axis[1, 2].set_title('weight change')
        axis[1, 2].set_xlabel('target [neuron id]')
        axis[1, 2].set_ylabel('sender [neuron id]')
        axis[1, 2].grid(False)
        


        figure.set_size_inches(17, 9)
        plt.tight_layout()
        plt.show()


    # def test_SynapseDepressionFacilitation(self):
    #     """Ensure that depression and facilitation work correctly"""

    #     nest.set_verbosity('M_WARNING')

    #     # This is done using the spike pairing experiment of
    #     # Clopath et al. 2010. First we specify the parameters
    #     resolution = 0.1
    #     init_w = 0.5
    #     spike_times_pre = [
    #         [29.,  129.,  229.,  329.,  429.],
    #         [29.,   62.3,   95.7,  129.,  162.3],
    #         [29.,   49.,   69.,   89.,  109.],
    #         [129.,  229.,  329.,  429.,  529.,  629.],
    #         [62.3,   95.6,  129.,  162.3,  195.6,  229.],
    #         [49.,   69.,   89.,  109.,  129.,  149.]]
    #     spike_times_post = [
    #         [19.,  119.,  219.,  319.,  419.],
    #         [19.,   52.3,   85.7,  119.,  152.3],
    #         [19.,  39.,  59.,  79.,  99.],
    #         [139.,  239.,  339.,  439.,  539.,  639.],
    #         [72.3,  105.6,  139.,  172.3,  205.6,  239.],
    #         [59.,   79.,   99.,  119.,  139.,  159.]]
    #     tested_models = ["aeif_psc_delta_clopath", "hh_psc_alpha_clopath"]

    #     # Loop over tested neuron models
    #     for nrn_model in tested_models:
    #         if(nrn_model == "aeif_psc_delta_clopath"):
    #             nrn_params = {'V_m': -70.6,
    #                           'E_L': -70.6,
    #                           'V_peak': 33.0,
    #                           'C_m': 281.0,
    #                           'theta_minus': -70.6,
    #                           'theta_plus': -45.3,
    #                           'A_LTD': 14.0e-5,
    #                           'A_LTP': 8.0e-5,
    #                           'tau_minus': 10.0,
    #                           'tau_plus': 7.0,
    #                           'delay_u_bars': 4.0,
    #                           'a': 4.0,
    #                           'b': 0.0805,
    #                           'V_reset': -70.6 + 21.0,
    #                           'V_clamp': 33.0,
    #                           't_clamp': 2.0,
    #                           't_ref': 0.0, }
    #         elif(nrn_model == "hh_psc_alpha_clopath"):
    #             nrn_params = {'V_m': -64.9,
    #                           'C_m': 100.0,
    #                           'tau_syn_ex': 0.2,
    #                           'tau_syn_in': 2.0,
    #                           'theta_minus': -64.9,
    #                           'theta_plus': -35.0,
    #                           'A_LTD': 14.0e-5,
    #                           'A_LTP': 8.0e-5,
    #                           'tau_minus': 10.0,
    #                           'tau_plus': 114.0,
    #                           'delay_u_bars': 5.0,
    #                           }
    #         syn_weights = []
    #         # Loop over pairs of spike trains
    #         for (s_t_pre, s_t_post) in zip(spike_times_pre, spike_times_post):
    #             nest.ResetKernel()
    #             nest.resolution = resolution

    #             # Create one neuron
    #             nrn = nest.Create(nrn_model, 1, nrn_params)
    #             prrt_nrn = nest.Create("parrot_neuron", 1)

    #             # Create and connect spike generator
    #             spike_gen_pre = nest.Create("spike_generator", 1, {
    #                                         "spike_times": s_t_pre})

    #             nest.Connect(spike_gen_pre, prrt_nrn,
    #                          syn_spec={"delay": resolution})

    #             if(nrn_model == "aeif_psc_delta_clopath"):
    #                 conn_weight = 80.0
    #             elif(nrn_model == "hh_psc_alpha_clopath"):
    #                 conn_weight = 2000.0

    #             spike_gen_params_post = {"spike_times": s_t_post}
    #             spike_gen_post = nest.Create("spike_generator", 1, {
    #                 "spike_times": s_t_post})

    #             nest.Connect(spike_gen_post, nrn, syn_spec={
    #                 "delay": resolution, "weight": conn_weight})

    #             # Create weight recorder
    #             wr = nest.Create('weight_recorder', 1)

    #             # Create Clopath synapse with weight recorder
    #             nest.CopyModel("clopath_synapse", "clopath_synapse_rec",
    #                            {"weight_recorder": wr})

    #             syn_dict = {"synapse_model": "clopath_synapse_rec",
    #                         "weight": init_w, "delay": resolution}
    #             nest.Connect(prrt_nrn, nrn, syn_spec=syn_dict)

    #             # Simulation
    #             simulation_time = (10.0 + max(s_t_pre[-1], s_t_post[-1]))
    #             nest.Simulate(simulation_time)

    #             # Evaluation
    #             w_events = nest.GetStatus(wr)[0]["events"]
    #             weights = w_events["weights"]
    #             syn_weights.append(weights[-1])

    #         # Compare to expected result
    #         syn_weights = np.array(syn_weights)
    #         syn_weights = 100.0*15.0*(syn_weights - init_w)/init_w + 100.0
    #         if(nrn_model == "aeif_psc_delta_clopath"):
    #             correct_weights = [57.82638722, 72.16730112, 149.43359357,
    #                                103.30408341, 124.03640668, 157.02882555]
    #         elif(nrn_model == "hh_psc_alpha_clopath"):
    #             correct_weights = [70.14343863, 99.49206222, 178.1028757,
    #                                119.63314118, 167.37750688, 178.83111685]

    #         self.assertTrue(np.allclose(
    #             syn_weights, correct_weights, rtol=1e-7))

def suite():
    suite = unittest.makeSuite(ClopathNeuronTestCase, 'test')
    return suite


def run():
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())


if __name__ == "__main__":
    run()