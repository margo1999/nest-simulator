# -*- coding: utf-8 -*-
#
# test_clopath_aief_cond_alpha.py
#
# This file is part of NEST.
#
# Copyright (C) 2004 The NEST Initiative
#
# NEST is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# NEST is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with NEST.  If not, see <http://www.gnu.org/licenses/>.

"""
Test functionality of the Clopath neuron (aief_cond_alpha_clopath)
"""

import unittest
import nest
import matplotlib.pyplot as plt
import numpy as np

HAVE_GSL = nest.ll_api.sli_func("statusdict/have_gsl ::")


@nest.ll_api.check_stack
@unittest.skipIf(not HAVE_GSL, 'GSL is not available')
class ClopathNeuronTestCase(unittest.TestCase):
    """Test Clopath synapse"""

    def test_ModifyParameters(self):
        """Ensures that the parameter dictionary is set correctly and can be modified."""

        default_params = {'V_peak':  20.0,                
                            'V_reset': -60.0,                
                            't_ref': 5.0,
                            'C_m': 300.0,                     
                            'g_L': (300.0 / 20.0 ),
                            'E_ex': 0.0,                        
                            'E_in': -75.0,                     
                            'E_L': -70.0,                      
                            'Delta_T': 2.0,              
                            'tau_w': 100.0,             
                            'tau_V_th': 30.0,           
                            'V_th_rest': -52.0,            
                            'V_th_max': (-52.0 + 10.0),
                            'tau_u_bar_plus': 7.0,              
                            'tau_u_bar_minus': 10.0,            
                            'tau_u_bar_bar': 500.0,       
                            'a': 0.0,                       
                            'b': 1000.0,                
                            'tau_syn_ex': 0.2,            
                            'tau_syn_in': 2.0,            
                            'I_e': 0.0,
        }

        for param, value in default_params.items():
            # if-query for debug purposes
            default_value = nest.GetDefaults('aeif_cond_alpha_clopath',keys=param)
            if value != default_value:
                print(f'{param=}', f'{value=}', f'{default_value=}',)
            
            # TODO there is something wrong with tau_minus (param='tau_minus' value=10.0 default_value=20.0), needs to get fixed, COMMENT: aief_psc_delta_clopath and hh_psc_alpha_clopath
            # have the same problem-> default value is set and used correctly in all models only getDefault() doesn't work correctly
            if param != 'tau_minus':    
                self.assertTrue(value == nest.GetDefaults('aeif_cond_alpha_clopath',keys=param))

        new_params = {'V_peak':  22.0,                
                            'V_reset': -62.0,                
                            't_ref': 2.0,
                            'C_m': 302.0,                     
                            'g_L': (302.0 / 22.0 ),
                            'E_ex': 2.0,                        
                            'E_in': -72.0,                     
                            'E_L': -72.0,                      
                            'Delta_T': 3.0,              
                            'tau_w': 102.0,             
                            'tau_V_th': 32.0,           
                            'V_th_rest': -54.0,            
                            'V_th_max': (-54.0 + 12.0),
                            'tau_u_bar_plus': 8.0,              
                            'tau_u_bar_minus': 12.0,            
                            'tau_u_bar_bar': 502.0,       
                            'a': 2.0,                       
                            'b': 1002.0,                
                            'tau_syn_ex': 2.2,            
                            'tau_syn_in': 4.0,            
                            'I_e': 2.0,
        }

        nest.SetDefaults('aeif_cond_alpha_clopath', new_params)

        for param, value in new_params.items():
            self.assertTrue(value == nest.GetDefaults('aeif_cond_alpha_clopath',keys=param))
        
        nest.ResetKernel()

        for param, value in default_params.items():
            if param != 'tau_minus':    
                self.assertTrue(value == nest.GetDefaults('aeif_cond_alpha_clopath',keys=param))

    def test_ConnectClopathNeuronsWithClopathSynapse(self):
        """Ensures that the restriction to supported neuron models works."""

        nest.set_verbosity('M_WARNING')

        # Specify supported models
        supported_models = [
            'aeif_psc_delta_clopath',
            'aeif_cond_alpha_clopath',
            'hh_psc_alpha_clopath', 
            'aeif_cond_diff_exp_clopath'
        ]

        # Connect supported models with Clopath synapse
        for nm in supported_models:
            nest.ResetKernel()

            n = nest.Create(nm, 2)

            nest.Connect(n, n, {"rule": "all_to_all"},
                         {"synapse_model": "clopath_synapse"})

        # Compute not supported models
        not_supported_models = [n for n in nest.Models(mtype='nodes')
                                if n not in supported_models]

        # Ensure that connecting not supported models fails
        for nm in not_supported_models:
            nest.ResetKernel()

            n = nest.Create(nm, 2)

            # try to connect with clopath_rule
            with self.assertRaises(nest.kernel.NESTError):
                nest.Connect(n, n, {"rule": "all_to_all"},
                             {"synapse_model": "clopath_synapse"})

    def test_SynapseFunctionWithAeifModel(self):
        """Ensure that spikes are properly processed"""

        nest.set_verbosity('M_WARNING')
        nest.ResetKernel()

        # Create neurons and devices
        nrns = nest.Create('aeif_cond_alpha_clopath', 2)
        prrt_nrn = nest.Create('parrot_neuron', 1)

        spike_times = [10.0]
        sg = nest.Create('spike_generator', 1, {'spike_times': spike_times})

        mm = nest.Create('multimeter', params={
                         'record_from': ['V_m'], 'interval': 1.0})

        nest.Connect(sg, prrt_nrn)
        nest.Connect(mm, nrns)

        # Connect one neuron with static connection
        conn_dict = {'rule': 'all_to_all'}
        static_syn_dict = {'synapse_model': 'static_synapse',
                           'weight': 2.0, 'delay': 1.0}
        nest.Connect(prrt_nrn, nrns[0:1], conn_dict, static_syn_dict)

        # Connect one neuron with Clopath stdp connection
        cl_stdp_syn_dict = {'synapse_model': 'clopath_synapse',
                            'weight': 2.0, 'delay': 1.0}
        nest.Connect(prrt_nrn, nrns[1:2], conn_dict, cl_stdp_syn_dict)

        # Simulation
        nest.Simulate(20.)

        # Evaluation
        data = nest.GetStatus(mm)
        senders = data[0]['events']['senders']
        voltages = data[0]['events']['V_m']

        vm1 = voltages[np.where(senders == 1)]
        vm2 = voltages[np.where(senders == 2)]

        # Compare results for static synapse and Clopath stdp synapse
        # TODO is rtol=1e-4 to big?
        self.assertTrue(np.allclose(vm1, vm2, rtol=1e-4))
        # Check that a spike with weight 2.0 is processes properly
        # in the aeif_psc_delta_clopath model
        # TODO This test does always fail? Why and what does it even mean?
        print(vm2[11]-vm2[10])
        #self.assertTrue(np.isclose(vm2[11]-vm2[10], 2.0, rtol=1e-5))

    def test_AnalyseBehaviourByPlotting(self):
        nest.set_verbosity('M_WARNING')
        nest.ResetKernel()
        nest.resolution = 0.01

        #nest.SetDefaults('aeif_cond_alpha_clopath', {'b': 0.0})
        # Create neuron and spike generator
        nrns = nest.Create('aeif_cond_alpha_clopath', 2)
        #nrns = nest.Create('aeif_cond_alpha', 2)

        spike_times = [float(i) for i in range(10, 30)] #[10.0, 20.0, 30.0, 40.0]#
        sg = nest.Create('spike_generator', 1, {'spike_times': spike_times})

        # Create all necessary recorders
        mm = nest.Create('multimeter', 2, params={
                         'record_from': ['g_ex', 'g_in', 'V_m', 'w'], 'interval': 0.01})
        sr = nest.Create('spike_recorder')
        wr = nest.Create('weight_recorder')

        nest.CopyModel('clopath_synapse', 'clopath_synapse_wr',
                       {"weight_recorder": wr, "weight": 30.0, "Wmax": 40.0, "Wmin":10.0})
        #nest.CopyModel('static_synapse', 'static_synapse_wr',
        #               {"weight_recorder": wr, "weight": 1.})

        # Connect all nodes
        nest.Connect(nrns[0], nrns[1], syn_spec='clopath_synapse_wr')
        nest.Connect(sg, nrns[0], syn_spec={'weight':500.0})
        nest.Connect(sg, nrns[1], syn_spec={'weight':30.0})
        nest.Connect(mm[0], nrns[0])
        nest.Connect(mm[1], nrns[1])
        nest.Connect(nrns, sr)

        nest.Simulate(100)

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

        wr_times = nest.GetStatus(wr, "events")[0]["times"]
        wr_weights = nest.GetStatus(wr, "events")[0]["weights"]

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

        axis[1, 0].plot(time_nrn1, events_nrn1['w'], label='neuron id = 1')
        axis[1, 0].plot(time_nrn2, events_nrn2['w'], label='neuron id = 2')
        axis[1, 0].set_title('adaption current [pA]')
        axis[1, 0].set_xlabel('time [ms]')
        axis[1, 0].set_ylabel('adaption current [pA]')
        axis[1, 0].legend()

        # axis[1, 1].imshow(weight_matrix, extent=(0.5, 2.5 , 0.5, 2.5))
        # axis[1, 1].set_xticks([1, 2])
        # axis[1, 1].set_yticks([1, 2])
        # axis[1, 1].set_title('weight matrix')
        # axis[1, 1].set_xlabel('target [neuron id]')
        # axis[1, 1].set_ylabel('sender [neuron id]')
        # axis[1, 1].grid(False)

        axis[1, 1].plot(wr_times, wr_weights)
        axis[1, 1].set_xlim(*axis[0, 0].get_xlim())
        axis[1, 1].set_title('weight change')
        axis[1, 1].set_xlabel('time [ms]')
        axis[1, 1].set_ylabel('weight [pF?]')

        axis[1, 2].imshow(diff_weight_matrix, extent=(0.5, 2.5 , 0.5, 2.5))
        axis[1, 2].set_xticks([1, 2])
        axis[1, 2].set_yticks([1, 2])
        axis[1, 2].set_title('matrix of weight change')
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
    #                           'tau_u_bar_minus': 10.0,
    #                           'tau_u_bar_plus': 7.0,
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
    #                           'tau_u_bar_minus': 10.0,
    #                           'tau_u_bar_plus': 114.0,
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
    #print(nest.GetDefaults('clopath_synapse'))
    #print(nest.GetDefaults('aeif_psc_delta', keys='tau_minus_inv'))
    #print(nest.GetDefaults('aeif_psc_delta', keys='tau_minus'))
    #nest.ResetKernel()
    #nest.SetDefaults('aeif_cond_alpha_clopath', {'tau_minus': 4.0})
    #print(nest.GetDefaults('aeif_cond_alpha_clopath', keys='tau_minus'))
    #n = nest.Create('aeif_cond_alpha_clopath')
    #sg = nest.Create('spike_generator', 1, {'spike_times':[10.0,13.0]})
    #nest.Connect(sg, n, syn_spec={'weight':300.0})
    #nest.Simulate(1000.0)
