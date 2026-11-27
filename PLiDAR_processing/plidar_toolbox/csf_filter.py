import CSF
import numpy as np


def cloth_simulation_filter(final_ground_building_data, cloth_resolution):
    x_y_z = np.vstack(
        (
            final_ground_building_data["x_act"],
            final_ground_building_data["y_act"],
            final_ground_building_data["h"],
        )
    ).T
    csf = CSF.CSF()
    csf.params.bSloopSmooth = False
    csf.params.cloth_resolution = cloth_resolution
    csf.setPointCloud(x_y_z)
    ground = CSF.VecInt()
    non_ground = CSF.VecInt()
    csf.do_filtering(ground, non_ground, False)

    ground = np.array(ground, dtype=int)
    non_ground = np.array(non_ground, dtype=int)

    filtered_data_ground = {
        "x_act": final_ground_building_data["x_act"][ground],
        "h": final_ground_building_data["h"][ground],
        "signal_conf_ph": final_ground_building_data["signal_conf_ph"][ground],
        "lat": final_ground_building_data["lat"][ground],
        "lon": final_ground_building_data["lon"][ground],
        "ref_elev": final_ground_building_data["ref_elev"][ground],
    }
    filtered_data_unground = {
        "x_act": final_ground_building_data["x_act"][non_ground],
        "h": final_ground_building_data["h"][non_ground],
        "signal_conf_ph": final_ground_building_data["signal_conf_ph"][non_ground],
        "lat": final_ground_building_data["lat"][non_ground],
        "lon": final_ground_building_data["lon"][non_ground],
        "ref_elev": final_ground_building_data["ref_elev"][non_ground],
    }
    return filtered_data_ground, filtered_data_unground
