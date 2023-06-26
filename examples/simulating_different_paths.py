import sys
sys.path.append('/Users/ludvigloite/projects/skole/prosjektoppgave/ship_simulator/')

from models import ShipModel, ShipConfiguration, EnvironmentConfiguration, \
    MachinerySystemConfiguration, SimulationConfiguration, \
    MachineryMode, MachineryModeParams, MachineryModes, ThrottleControllerGains, \
    EngineThrottleFromSpeedSetPoint, HeadingByRouteController, HeadingControllerGains, \
    SpecificFuelConsumptionWartila6L26, SpecificFuelConsumptionBaudouin6M26Dot3, LosParameters, \
    SeachartSimulationConfiguration
import numpy as np
import shapely.geometry as geo
from random import random
from tqdm import tqdm
import pickle
from datetime import date
import os

import pysmile
import pysmile_license

probabilityNet = pysmile.Network()
probabilityNet.read_file("probabilityBN.xdsl")

damageStateNet = pysmile.Network()
damageStateNet.read_file("DamageStateBN.xdsl")


ship_info = {
    'ship_type': 'Passenger',
    'number_of_tanks': 12,
    'tank_size': 5000,
    'value': 2150000, #From shipselector.com 87m roro ship
    'oil_price': 1000,
    'daily_profit': 1000,
    'blackout_base_probability': 1.3*10**(-8) * 2
}

cost_function_weights = {
    'alpha_env': 1,
    'alpha_soc': 1,
    'alpha_eco': 1
}


simulation_locations = ["tautra_north_1", "tautra_north_2", "tautra_north_3","tautra_south_1", "tautra_south_2", "tautra_south_3", "sekken_north_1", "sekken_south_1", "sekken_south_2", "sekken_south_3", "sandoya_east_1", "sandoya_west_1"]

sim_location = "tautra_north_3"
new_data = True
simulate_failures = True

simulation_locations_used = [i for i in simulation_locations if sim_location in i]

show_seacharts=False
show_plots=False
save_results_to_csv=False
risky_result_svg_name=''
draw_failure_ships = False
draw_safety_margin_violations = False

draft_of_ship=6
safety_margin=50

today = date.today()
folder_name='examples/reports/test_'+str(today)
svg_folder_name = 'test_'+str(today)
try:
    os.mkdir(folder_name)
except OSError:
    print ("Creation of the directory %s failed" % folder_name)
else:
    print ("Successfully created the directory %s " % folder_name)

try:
    os.mkdir('reports/'+svg_folder_name)
except OSError:
    print ("Creation of the directory %s failed" % 'reports/'+svg_folder_name)
else:
    print ("Successfully created the directory %s " % 'reports/'+svg_folder_name)


duration_of_failure_seconds = 0
duration_of_failure_simulation_seconds = 0

env_config=EnvironmentConfiguration(
current_velocity_component_from_north= -3,
current_velocity_component_from_east= 0,
wind_speed=0,
wind_direction=0)

desired_forward_speed_meters_per_second = 8.5
meters_between_each_failure_simulation = 20
time_since_last_ship_drawing = 30
duration_of_failure_seconds = 300
duration_of_failure_simulation_seconds = 300
time_step = 0.5

failure_simulation_time_interval = int(meters_between_each_failure_simulation / (desired_forward_speed_meters_per_second*time_step))


for simulation_location in simulation_locations_used:
    print("Simulating ",simulation_location," with failures lasting ",duration_of_failure_simulation_seconds," seconds")

    route_txt = "examples/route_seacharts.txt"
    conservation_areas_txt = 'examples/conservation_areas_standard.txt'
    init_north_pos = 6955000
    init_east_pos = 33100
    size_map_east = 18000
    size_map_north = 10124
    center_map_east = 36580
    center_map_north = 6960000
    init_yaw_angle = 90 * np.pi / 180

    route_txt = 'examples/route_'+simulation_location+'.txt'

    simulation_area_name = simulation_location.split('_')[0]
    conservation_areas_txt = 'examples/conservation_areas_'+simulation_area_name+'.txt'
    
    if simulation_area_name == "tautra":
        init_north_pos = 6974308
        init_east_pos = 81150
        size_map_east = 18000
        size_map_north = 10124
        center_map_east = 85814
        center_map_north = 6974797
        init_yaw_angle = 90 * np.pi / 180
        
        simulation_time = 1500

        env_config=EnvironmentConfiguration(
        current_velocity_component_from_north= -4,
        current_velocity_component_from_east= 1,
        wind_speed=0,
        wind_direction=0)

        probabilityNet.set_evidence("Current_Evaluation","Harsh")
        probabilityNet.set_evidence("Wind_Evaluation","Harsh") 
        probabilityNet.set_evidence("Flag_of_Registery","Norway") 
        probabilityNet.set_evidence("Vessel_Age","older_than_20")
        probabilityNet.set_evidence("Vessel_Type","Passenger")

        damageStateNet.set_evidence("Weather","Poor") 
        damageStateNet.set_evidence("Speed","Over_15_knots")
        damageStateNet.set_evidence("Ground_material","Rocks")

        probabilityNet.update_beliefs()
        blackoutUtily = probabilityNet.get_node_value("Blackout_utility_metric") 

        damageStateNet.update_beliefs()
        damageState = damageStateNet.get_node_value("Damage_State")
    
    elif simulation_area_name == "sekken":
        init_north_pos = 6972461
        init_east_pos = 98909
        size_map_east = 18000
        size_map_north = 10124
        center_map_east = 106405 
        center_map_north = 6971573
        init_yaw_angle = 90 * np.pi / 180

        simulation_time = 2500

        env_config=EnvironmentConfiguration(
        current_velocity_component_from_north= 0,
        current_velocity_component_from_east= 4,
        wind_speed=0,
        wind_direction=0)

        probabilityNet.set_evidence("Current_Evaluation","Harsh")
        probabilityNet.set_evidence("Wind_Evaluation","Harsh") 
        probabilityNet.set_evidence("Flag_of_Registery","Norway") 
        probabilityNet.set_evidence("Vessel_Age","older_than_20")
        probabilityNet.set_evidence("Vessel_Type","Passenger")

        damageStateNet.set_evidence("Weather","Poor") 
        damageStateNet.set_evidence("Speed","Over_15_knots")
        damageStateNet.set_evidence("Ground_material","Rocks")

        probabilityNet.update_beliefs()
        blackoutUtily = probabilityNet.get_node_value("Blackout_utility_metric") 
           
        damageStateNet.update_beliefs()
        damageState = damageStateNet.get_node_value("Damage_State")

    
    elif simulation_area_name == "sandoya":
        init_north_pos = 6948058
        init_east_pos = 23147
        size_map_east = int(18000*0.6)
        size_map_north = int(10124*0.6)
        center_map_east = 24470 
        center_map_north = 6950311
        init_yaw_angle = 0 * np.pi / 180
        init_yaw_angle = 30 * np.pi / 180


        env_config=EnvironmentConfiguration(
        current_velocity_component_from_north= -4,
        current_velocity_component_from_east= 0,
        wind_speed=0,
        wind_direction=0)

        simulation_time = 1000

        ship_info = {
            'ship_type': 'Oil_tanker',
            'number_of_tanks': 12,
            'tank_size': 5000,
            'value': 2000000, #From shipselector.com 87m roro ship
            'oil_price': 1000,
            'daily_profit': 10000,
            'blackout_base_probability': 1.3*10**(-8) * 2
        }

        probabilityNet.set_evidence("Current_Evaluation","Moderate")
        probabilityNet.set_evidence("Wind_Evaluation","Moderate") 
        probabilityNet.set_evidence("Flag_of_Registery","Norway") 
        probabilityNet.set_evidence("Vessel_Age","older_than_20")
        probabilityNet.set_evidence("Vessel_Type","Oil_Tanker")

        probabilityNet.update_beliefs()
        blackoutUtily = probabilityNet.get_node_value("Blackout_utility_metric")


        damageStateNet.set_evidence("Weather","Normal") 
        damageStateNet.set_evidence("Speed","Over_15_knots")
        damageStateNet.set_evidence("Ground_material","Rocks")

        damageStateNet.update_beliefs()
        damageState = damageStateNet.get_node_value("Damage_State")
    




    main_engine_capacity = 2160e3
    diesel_gen_capacity = 510e3
    hybrid_shaft_gen_as_generator = 'GEN'
    hybrid_shaft_gen_as_motor = 'MOTOR'
    hybrid_shaft_gen_as_offline = 'OFF'

    # Configure the simulation
    ship_config = ShipConfiguration(
        coefficient_of_deadweight_to_displacement=0.7,
        bunkers=20000,
        ballast=20000,
        length_of_ship=80,
        width_of_ship=16,
        added_mass_coefficient_in_surge=0.4,
        added_mass_coefficient_in_sway=0.4,
        added_mass_coefficient_in_yaw=0.4,
        dead_weight_tonnage=3850000,
        mass_over_linear_friction_coefficient_in_surge=130,
        mass_over_linear_friction_coefficient_in_sway=18,
        mass_over_linear_friction_coefficient_in_yaw=90,
        nonlinear_friction_coefficient__in_surge=2400,
        nonlinear_friction_coefficient__in_sway=4000,
        nonlinear_friction_coefficient__in_yaw=400
    )
    mec_mode_params = MachineryModeParams(
        main_engine_capacity=main_engine_capacity,
        electrical_capacity=diesel_gen_capacity,
        shaft_generator_state=hybrid_shaft_gen_as_offline
    )
    mec_mode = MachineryMode(params=mec_mode_params)
    mso_modes = MachineryModes(
        [mec_mode]
    )
    fuel_spec_me = SpecificFuelConsumptionWartila6L26()
    fuel_spec_dg = SpecificFuelConsumptionBaudouin6M26Dot3()
    machinery_config = MachinerySystemConfiguration(
        machinery_modes=mso_modes,
        machinery_operating_mode=0,
        linear_friction_main_engine=68,
        linear_friction_hybrid_shaft_generator=57,
        gear_ratio_between_main_engine_and_propeller=0.6,
        gear_ratio_between_hybrid_shaft_generator_and_propeller=0.6,
        propeller_inertia=6000,
        propeller_diameter=3.1,
        propeller_speed_to_torque_coefficient=7.5,
        propeller_speed_to_thrust_force_coefficient=1.7,
        hotel_load=200000,
        rated_speed_main_engine_rpm=1000,
        rudder_angle_to_sway_force_coefficient=50e3,
        rudder_angle_to_yaw_force_coefficient=500e3,
        max_rudder_angle_degrees=30,
        specific_fuel_consumption_coefficients_me=fuel_spec_me.fuel_consumption_coefficients(),
        specific_fuel_consumption_coefficients_dg=fuel_spec_dg.fuel_consumption_coefficients()
    )
    simulation_setup = SimulationConfiguration(
        initial_north_position_m= init_north_pos,
        initial_east_position_m= init_east_pos,
        initial_yaw_angle_rad=init_yaw_angle,
        initial_forward_speed_m_per_s=7,
        initial_sideways_speed_m_per_s=0,
        initial_yaw_rate_rad_per_s=0,
        integration_step=time_step,
        simulation_time=simulation_time,
    )
    seacharts_setup = SeachartSimulationConfiguration(
        size_of_map_east = size_map_east,
        size_of_map_north = size_map_north,
        center_of_map_east = center_map_east,
        center_of_map_north = center_map_north,
        database_file_names = ['More_og_Romsdal.gdb'],
        new_data = new_data,
        border = True,
        route_txt = route_txt,
        waypoint_color = 'green',
        draft_of_ship = draft_of_ship, #Typical draft around 6m: https://horizonship.com/ship/80m-dp2-platform-supply-vessel-1-of-3-sister-ships-2020-dwt-3500/
        safety_margin = safety_margin,
        wind_arrow_drawing_coefficient = 20,
        current_arrow_drawing_coefficient = 1000,
        current_wind_arrow_offset = 500,
        wind_arrow_color = 'black',
        current_arrow_color = 'yellow',
        able_to_crash = True,
        conservation_areas_txt=conservation_areas_txt
    )
    ship_model = ShipModel(ship_config=ship_config,
                        machinery_config=machinery_config,
                        environment_config=env_config,
                        seachart_config=seacharts_setup,
                        simulation_config=simulation_setup,
                        initial_propeller_shaft_speed_rad_per_s=400 * np.pi / 30)


    # Set up control systems
    throttle_controller_gains = ThrottleControllerGains(
        kp_ship_speed=7, ki_ship_speed=0.13, kp_shaft_speed=0.05, ki_shaft_speed=0.005
    )
    throttle_controller = EngineThrottleFromSpeedSetPoint(
        gains=throttle_controller_gains,
        max_shaft_speed=ship_model.ship_machinery_model.shaft_speed_max,
        time_step=time_step,
        initial_shaft_speed_integral_error=114
    )

    heading_controller_gains = HeadingControllerGains(kp=4, kd=90, ki=0.01)
    los_guidance_parameters = LosParameters(
        radius_of_acceptance=200,
        lookahead_distance=500,
        integral_gain=0.002,
        integrator_windup_limit=4000
    )
    auto_pilot = HeadingByRouteController(
        route_name=route_txt,
        heading_controller_gains=heading_controller_gains,
        los_parameters=los_guidance_parameters,
        time_step=time_step,
        max_rudder_angle=machinery_config.max_rudder_angle_degrees * np.pi/180
    )

    ship_color = 'orange'

    integrator_term = []
    times = []
    failures = []

    #These may be used by others, but now only FM1 is simulated
    colors = {'FM1: Full Blackout':'red', 'FM2: 80% Power Loss':'green', 'FM3: 50% Power Loss':'yellow', 'FM4: Rudder Freeze':'black'}
    failure_modes= ['FM1: Full Blackout', 'FM2: 80% Power Loss', 'FM3: 50% Power Loss', 'FM4: Rudder Freeze']

    pbar = tqdm(total=simulation_time)

    while ship_model.int.time < ship_model.int.sim_time and auto_pilot.next_wpt != auto_pilot.prev_wpt:
        
        pbar.update(time_step)

        # Measure position and speed
        north_position = ship_model.north
        east_position = ship_model.east
        heading = ship_model.yaw_angle
        speed = ship_model.forward_speed

        

        # Find appropriate rudder angle and engine throttle
        rudder_angle = auto_pilot.rudder_angle_from_route(
            north_position=north_position,
            east_position=east_position,
            heading=heading
        )

        throttle = throttle_controller.throttle(
            speed_set_point=desired_forward_speed_meters_per_second,
            measured_speed=speed,
            measured_shaft_speed=speed
        )
       
        #Failure simulation     
        if simulate_failures and ship_model.int.time % failure_simulation_time_interval == 0:
            throttle_controller, auto_pilot, failures = ship_model.simulate_failure(failure_type='FM1: Full Blackout', nu_of_timesteps=duration_of_failure_simulation_seconds, failure_time_length=duration_of_failure_seconds, auto_pilot=auto_pilot, throttle_controller=throttle_controller, desired_forward_speed_meters_per_second=desired_forward_speed_meters_per_second, ship_color=colors['FM1: Full Blackout'], rudder_angle=rudder_angle, throttle=throttle, failures=failures, printAllFailures=False, draw_ships=draw_failure_ships, probabilityNet=probabilityNet, damageStateNet=damageStateNet, ship_info=ship_info,cost_function_weights=cost_function_weights, draw_safety_margin_violations = draw_safety_margin_violations)
            
        # Consequence Level Evaluation
        ship_color = 'white'

        position = geo.Point(east_position,north_position)
        dist_land = position.distance(ship_model.landGeometry)
        dist_shore = position.distance(ship_model.shoreGeometry)

        minimum_distance_to_hazard = min(dist_land, dist_shore)

        for i in range(len(ship_model.depths)):
            if ship_model.depths[i] <= ship_model.draft_of_ship:
                dist_seabed = position.distance(ship_model.seabedList[ship_model.depths[i]])
                if dist_seabed < minimum_distance_to_hazard:
                    minimum_distance_to_hazard = dist_seabed

        if minimum_distance_to_hazard <= ship_model.safety_margin:
            ship_color = 'orange'
        if minimum_distance_to_hazard == 0 and ship_model.able_to_crash:
            ship_model.add_vessel_drawing(east_position=east_position, north_position=north_position, heading=heading, ship_color=ship_color)
            #ship_model.enc.draw_circle((east_position,north_position), 50, 'red', thickness=10, fill=False)
            break

        # Update and integrate differential equations for current time step
        ship_model.store_simulation_data(throttle, rudder_angle)

        ship_model.update_differentials(engine_throttle=throttle, rudder_angle=rudder_angle)
        ship_model.integrate_differentials()

        integrator_term.append(auto_pilot.navigate.e_ct_int)
        times.append(ship_model.int.time)

        # Make a drawing of the ship from above every 20 second
        if time_since_last_ship_drawing > 30 or auto_pilot.next_wpt == auto_pilot.prev_wpt:
            ship_model.add_vessel_drawing(east_position=east_position, north_position=north_position, heading=heading, ship_color=ship_color)
            ship_model.ship_snap_shot()
            time_since_last_ship_drawing = 0
        time_since_last_ship_drawing += ship_model.int.dt

        # Progress time variable to the next time step
        ship_model.int.next_time()
        
    pbar.close()

    count_crash = 0
    count_safety_margin = 0
    total_consequence_level = 0
    total_risk = 0
    total_consequences= {'envConsequence' : 0, 'socialConsequence' : 0, 'economicConsequence' : 0}
    total_damage_states = [0 for i in range(6)]

    for element in ship_model.consequence_dicts:
        total_consequence_level = total_consequence_level + element['total_consequence']
        total_risk = total_risk + element["risk"]
        total_consequences['envConsequence'] += element['consequences']['envConsequence']
        total_consequences['socialConsequence'] += element['consequences']['socialConsequence']
        total_consequences['economicConsequence'] += element['consequences']['economicConsequence']
       
        total_damage_states = [x + y for x, y in zip(total_damage_states, element['damage_state'])]


        if element['worst_violation']=='did_crash':
            count_crash += 1
        elif element['worst_violation']=='did_violate_safety_margin':
            count_safety_margin += 1
    
    result_dict = {}
    result_dict['route'] = simulation_location
    result_dict['duration_of_failure_seconds'] = duration_of_failure_seconds
    result_dict['duration_of_failure_simulation_seconds'] = duration_of_failure_simulation_seconds
    result_dict['fuel_consumption_kgs'] = ship_model.simulation_results['fuel consumption [kg]'][-1]
    result_dict['time_usage_seconds'] = ship_model.int.time
    result_dict['nu_of_crashes'] = count_crash
    result_dict['nu_of_safety_margin_violation'] = count_safety_margin
    result_dict['total_consequence_level'] = total_consequence_level
    result_dict['total_risk'] = total_risk
    result_dict['total_consequences'] = total_consequences
    result_dict['total_damage_states'] = total_damage_states
    result_dict['consequence_dicts'] = ship_model.consequence_dicts

    with open(folder_name+'/'+'result_dict_'+str(simulation_location)+'_'+str(duration_of_failure_seconds)+'sec_'+str(duration_of_failure_simulation_seconds)+'sec.pkl', 'wb') as f:
        pickle.dump(result_dict, f)


    ship_model.enc.add_vessels(*ship_model.ship_positions)
    ship_model.draw_wind_and_current_arrows()
    ship_model.enc.save_image(name=svg_folder_name+'/'+str(simulation_location), extension='svg')
    #ship_model.enc.show_display()

