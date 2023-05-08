import firing_rate
import subjects

subject = subjects.subjects['18-1']
data = subjects.load_data(subject, units=True)

unit_id, strengths, modulations = firing_rate.swr_modulated(data['hab']['int_units'], data['hab']['sample_rate'],
                                                            ripple_windows=data['hab']['Sleep1']['ripple_windows'],
                                                            ts=data['hab']['Sleep1']['ts'], spike_windows=[])