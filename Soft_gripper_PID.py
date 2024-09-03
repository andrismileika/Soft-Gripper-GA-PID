import os
import json
import numpy as np
from simple_pid import PID
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
import time
import numbers
import pyfirmata
import pyfirmata.util as util
import pandas as pd

import pygad
from filterpy.kalman import KalmanFilter

PRESSURE_BLEED_COEF = 15
MAX_PRESSURE_CHANGE_COEF = 0.0005
MAX_PRESSURE_CHANGE_POWER = 1.25

SETTLING_TIME_FUTURE_SECONDS = 3
STEADY_STATE_ALLOWED_ERROR = 2
RISE_TIME_ALLOWED_ERROR = 0.1

w_iae = 3
w_rs = 1
w_st = 2
w_os = 3
w_sse = 2

class SoftGripper:
    def __init__(self, data_path:str="./data", dt:float=0.05, sim_time:float=10, kalman_filter=True):
        self.hyperbolic_coefs = {}
        self.__load_data(data_path)

        self.dt = dt
        self.sim_time = sim_time

        self.pwm_lower_limit = 80
        self.pwm_upper_limit = 255
        self.num_steps = int(self.sim_time / self.dt)

        self.IAE = None
        self.rise_time = None
        self.settling_time = None
        self.overshoot = None
        self.steady_state_error = None
        
        self.__arduino_connected = False

        self.motor_pwm = None
        self.solenoid = None
        self.pressure_sensor = None

        self.activate_kalman_filter = kalman_filter
        if kalman_filter:
            kalman_filter = SoftGripperKalmanFilter()
            self.__kf = kalman_filter.kf

        self.iteration_data = pd.DataFrame(
            [],
            columns=["SP", "Population size", "Mutation rate", "Crossover rate", "Number of generations", "IAE", "Rise time", "Settling time", "Overshoot", "SSE", "Fitness", "P", "I", "D"]
        )
        self.average_data = pd.DataFrame(
            [],
            columns=["SP", "Population size", "Mutation rate", "Crossover rate", "Number of generations", "IAE", "Rise time", "Settling time", "Overshoot", "SSE", "Average fitness"]
        )
        
        
        
        
    def __load_data(self, path:str) -> None:
        data_files = os.listdir(path)
        for data_file in data_files:

            if not data_file.startswith("pwm_hyperbolic_coefficients_"):
                continue
            
            file = open(os.path.join(path,data_file), 'r')
            data = json.load(file)

            pwm_key = list(data.keys())[0]
            pwm_val = int(pwm_key)
            a = float(data[pwm_key]["a"])
            b = float(data[pwm_key]["b"])
            c = float(data[pwm_key]["c"])
            
            self.hyperbolic_coefs[pwm_val] = {}
            self.hyperbolic_coefs[pwm_val]["a"] = a
            self.hyperbolic_coefs[pwm_val]["b"] = b
            self.hyperbolic_coefs[pwm_val]["c"] = c
        
            file.close()

    def reset_kalman_filter(self):
        if self.activate_kalman_filter:
            del self.__kf
            kalman_filer = SoftGripperKalmanFilter()
            self.__kf = kalman_filer.kf

    def pid_value_test(self, P:float, I:float, D:float, SP:float=40.0, save_to_file=False, ga_params=None, plot_results=True, save_image=False, best_fitness=None):
        
        pid = PID(P, I, D, setpoint=SP, sample_time=None)
        pid.output_limits = (self.pwm_lower_limit, self.pwm_upper_limit)

        time_values = []
        pressure_values = []
        pressure_kalman_values = []

        current_time = 0
        current_pressure = 0
        
        for _ in range(self.num_steps):
                
            pwm = pid(current_pressure, dt=self.dt)
            pwm = round(pwm)
                       
            current_pressure = self.motor_pressure_response(pwm, current_time, current_pressure)

            if self.activate_kalman_filter:
                self.__kf.predict()
                self.__kf.update(current_pressure)
                pressure_kalman_values.append(self.__kf.x[0])
                        
            time_values.append(current_time)
            pressure_values.append(current_pressure)

            current_time += self.dt

        time_values = np.array(time_values)
        pressure_values = np.array(pressure_values)


        if self.activate_kalman_filter:
            pressure_kalman_values = np.array(pressure_kalman_values)
        
        self.__print_metrics(SP, time_values, pressure_values, pressure_kalman_values, pid=(P, I, D),save_to_file=save_to_file, ga_params=ga_params, best_fitness=best_fitness)
        self.__plot_results((P, I, D), SP, time_values, pressure_values, pressure_kalman_values, plot_results=plot_results, save_image=save_image, ga_params=ga_params)
    
    def motor_pressure_response(self, pwm:int, t:float, current_pressure:float):
        """
        Simulates the pressure response of the motor given the PWM input.
        """
        pressure = current_pressure
        if pwm != 0:
            a, b, c = self.hyperbolic_coefs[pwm].values()
            new_pressure = self.hyperbolic_func(t, a, b, c)

            pressure_diff = new_pressure - current_pressure

            max_pressure_increase_in_dt = abs(MAX_PRESSURE_CHANGE_COEF * pressure_diff * pwm**MAX_PRESSURE_CHANGE_POWER) # kPa
            
            if pressure_diff > 0 :
                pressure = current_pressure + min(pressure_diff, max_pressure_increase_in_dt)
            else:
                pressure = current_pressure + max(pressure_diff, -max_pressure_increase_in_dt)

        pressure_bleed = current_pressure/PRESSURE_BLEED_COEF * self.dt
        pressure -= pressure_bleed

        return pressure
    
    def math_model_pressure_bleed_test(self, SP, duration, dt):
        time_vector = []
        pressure_vector = []

        elapsed_time = 0
        pwm = 255
        pressure = 0
        while len(time_vector) < duration/dt +1:
            pressure = self.motor_pressure_response(pwm, elapsed_time, pressure)

            if pressure >= SP:
                pwm = 0

            time_vector.append(elapsed_time)
            pressure_vector.append(pressure)
            elapsed_time += dt
        
        
        time_values = np.array(time_vector)
        pressure_values = np.array(pressure_vector)

        plt.figure(figsize=(10, 5))
        plt.plot(time_values, pressure_values, label='Spiediens')
        plt.axhline(y=SP, color='r', linestyle='--', label='SP')
        plt.ylim([0, SP+10])
        plt.xlabel('Laiks, s')
        plt.ylabel('Spiediens, kPa')
        plt.title(f'Pressure bleed (motor turn off at SP)')
        plt.legend()
        plt.grid(True)
        plt.show()

    def hyperbolic_func(self, t, a, b, c):
        val = a / (-t - b) + c
        return max(val, 0)

    def get_IAE(self, time_vector, pressure_vector, dt, SP):
        errors = SP - pressure_vector
        total_errors = np.abs(errors) * dt
        self.IAE = np.round(np.sum(total_errors), 4)
        return self.IAE
    
    def get_rise_time(self, time_vector, pressure_vector, SP):
        rise_time_vec = time_vector[pressure_vector >= SP-RISE_TIME_ALLOWED_ERROR]
        if len(rise_time_vec):
            self.rise_time = np.round(rise_time_vec[0], 4)
        else:
            self.rise_time = None
        return self.rise_time
    
    def get_settling_time(self, time_vector, pressure_vector, dt, SP):

        number_of_elems = int(SETTLING_TIME_FUTURE_SECONDS / dt)

        for i in range(len(time_vector)):
            next_pressures = pressure_vector[i:i+number_of_elems]
            vector_len = len(next_pressures)
            if vector_len != number_of_elems:
                break
                
            for j in range(vector_len):
                if abs(next_pressures[j]-SP) > STEADY_STATE_ALLOWED_ERROR:
                    break
            
            if j == vector_len - 1:

                self.settling_time = np.round(time_vector[i], 4)
                return self.settling_time
            
        return None
    
    def get_overshoot(self, time_vector, pressure_vector, SP):
        self.overshoot = np.round(np.max(pressure_vector) - SP, 4)
        if self.overshoot < 0:
            self.overshoot = 0

        return self.overshoot
    
    def get_steady_state_error(self, time_vector, pressure_vector, SP):
        if self.settling_time == None:
            return None
        
        self.steady_state_error = np.round(np.mean(np.abs(pressure_vector[time_vector >= self.settling_time] - SP)), 4)
           
        return self.steady_state_error
    
    def __print_metrics(self, SP, time_values, pressure_values, pressure_kalman_values=None, save_to_file=False, pid=None, ga_params=None, best_fitness=None, real_gripper=False):

        IAE = self.get_IAE(time_values, pressure_values, self.dt, SP)
        IAE_kalman = self.get_IAE(time_values, pressure_kalman_values, self.dt, SP) if self.activate_kalman_filter else None
        rise_time = self.get_rise_time(time_values, pressure_values, SP)
        rise_time_kalman = self.get_rise_time(time_values, pressure_kalman_values, SP) if self.activate_kalman_filter else None
        settling_time = self.get_settling_time(time_values, pressure_values, self.dt, SP)
        settling_time_kalman = self.get_settling_time(time_values, pressure_kalman_values, self.dt, SP) if self.activate_kalman_filter else None
        steady_state_error = self.get_steady_state_error(time_values, pressure_values, SP)
        steady_state_error_kalman = self.get_steady_state_error(time_values, pressure_kalman_values, SP)
        overshoot = self.get_overshoot(time_values, pressure_values, SP)
        overshoot_kalman = self.get_overshoot(time_values, pressure_kalman_values, SP) if self.activate_kalman_filter else None
        
        IAE_kalman_str = f" | IAE (with Kalman): {IAE_kalman}" if self.activate_kalman_filter else ""
        rise_time_kalman_str = f" | Rise time (with Kalman): {rise_time_kalman}" if self.activate_kalman_filter else ""
        settling_time_kalman_str = f" | Settling time (with Kalman): {settling_time_kalman}" if self.activate_kalman_filter else ""
        if self.activate_kalman_filter:
            overshoot_kalman_str = f" | Overshoot (with Kalman): {overshoot_kalman} {overshoot_kalman/SP*100} %)"
        else:
            overshoot_kalman_str = ""
        sse_kalman_str = f" | Steady state error (with Kalman): {steady_state_error_kalman}" if self.activate_kalman_filter else ""

        
        out_str = "======= METRICS =======\n" + \
        f"IAE: {IAE}{IAE_kalman_str}\n" + \
        f"Rise time: {rise_time}{rise_time_kalman_str}\n" + \
        f"Settling time: {settling_time}{settling_time_kalman_str}\n" + \
        f"Overshoot: {overshoot} ({overshoot/SP*100} %){overshoot_kalman_str}\n" + \
        f"Steady state error: {steady_state_error}{sse_kalman_str}\n" +\
        "======================="

        print(out_str)

        if not save_to_file:
            return
        P, I, D = pid
        if not real_gripper:
            sol_per_pop, mut_prob, cross_prob, num_gen = ga_params
        folder = "results_real_gripper" if real_gripper else "results"
        os.makedirs(folder, exist_ok=True)
        if real_gripper:

            file_name_all, ext = f"results_real_all", ".xlsx"
        else:
            file_name_all, ext = f"results_sim_all", ".xlsx"
            file_name_avg, ext = f"results_sim_average", ".xlsx"
        
        
        if not real_gripper and ga_params:
           
            self.iteration_data.loc[len(self.iteration_data)] = [
                SP,
                sol_per_pop, # population size
                mut_prob, #mutation rate
                cross_prob, #crossover rate
                num_gen, # Number of generations
                IAE_kalman if IAE_kalman != None else IAE,
                rise_time_kalman if rise_time_kalman != None else rise_time,
                settling_time_kalman if settling_time_kalman!=None else settling_time,
                overshoot_kalman if overshoot_kalman != None else overshoot,
                steady_state_error_kalman if steady_state_error_kalman else steady_state_error,
                best_fitness if best_fitness else -1,
                P, I, D
            ]
            self.iteration_data.to_excel(
                os.path.join(folder, file_name_all+ext)
            )
            
            matching_iters =self.iteration_data[
                (self.iteration_data["Population size"] == sol_per_pop) &
                (self.iteration_data["Mutation rate"] == mut_prob) &
                (self.iteration_data["Crossover rate"] == cross_prob) &
                (self.iteration_data["Number of generations"] == num_gen)
            ]
            iteration_nr = len(matching_iters)

            if iteration_nr > 0 and iteration_nr % 3 == 0:
                mean_metrics = matching_iters[["IAE", "Rise time", "Settling time", "Overshoot", "SSE", "Fitness"]].mean(numeric_only=True)

                self.average_data.loc[len(self.average_data)] = [
                    SP,
                    sol_per_pop, # population size
                    mut_prob, #mutation rate
                    cross_prob, #crossover rate
                    num_gen, # Number of generations
                    mean_metrics["IAE"],
                    mean_metrics["Rise time"],
                    mean_metrics["Settling time"],
                    mean_metrics["Overshoot"],
                    mean_metrics["SSE"],
                    mean_metrics["Fitness"]
                ]
                self.average_data.to_excel(
                    os.path.join(folder, file_name_avg+ ext)
                )
        elif real_gripper:
            try:
                real_data = pd.read_excel(os.path.join(folder, file_name_all + ext))
            except:

                self.real_data = pd.DataFrame(
                    [],
                    columns=["P", "I", "D", "IAE", "Rise time", "Settling time", "Overshoot", "SSE"]
                )
                
            self.real_data.loc[len(self.real_data)] = [
                P, I, D,
                IAE_kalman if IAE_kalman != None else IAE,
                rise_time_kalman if rise_time_kalman != None else rise_time,
                settling_time_kalman if settling_time_kalman!=None else settling_time,
                overshoot_kalman if overshoot_kalman != None else overshoot,
                steady_state_error_kalman if steady_state_error_kalman else steady_state_error
            ]
            
            self.real_data.to_excel(
               os.path.join(folder, file_name_all+ext) 
            )
            
   
    def __plot_results(self, pid, SP, time_values, pressure_values, pressure_kalman_values=None, real_gripper=False, plot_results=True, save_image=False, ga_params=None):
        plt.figure(figsize=(10, 5))
        if np.all(pressure_values!=pressure_kalman_values):
            plt.plot(time_values, pressure_values, label='Spiediens')
        if self.activate_kalman_filter:
            plt.plot(time_values, pressure_kalman_values, c='g', label="Spiediens ar Kalmana filtru", linewidth=2.5)
        plt.axhline(y=SP, color='r', linestyle='--', label='SP')
        plt.ylim([0, SP+10])
        plt.xlabel('Laiks, s')
        plt.ylabel('Spiediens, kPa')
        end_str = ' uz reālā satvērēja.' if real_gripper else '.'
        plt.title(f'Spiediena-laika līkne pie P={pid[0]:.4f}, I={pid[1]:.4f}, D={pid[2]:.4f}{end_str}')
        plt.legend()
        plt.grid(True)

        if save_image:
            folder = f"real_gripper_results_SP_{SP}" if real_gripper else f"results_SP_{SP}"
            os.makedirs(folder, exist_ok=True)
            if not real_gripper:
                sol_per_pop, mut_prob, cross_prob, num_gen = ga_params
                file_name, ext = f"results_spp{sol_per_pop}_mutRate{mut_prob}_crossRate{cross_prob}_numGen{num_gen}", ".png"
            else :
                P,I,D= pid
                file_name, ext = f"results_P_{P}_I_{I}_D_{D}_SP_{SP}", ".png"
            existing_files = os.listdir(folder)
            existing_files_w_same_params = list(filter(lambda x: (x.startswith(file_name) and x.endswith(ext)), existing_files))
            new_file_id = f"___{len(existing_files_w_same_params)+1}"
            plt.savefig(os.path.join(folder, file_name+new_file_id+ext))
            print(f"Saved image to {os.path.join(folder, file_name+new_file_id+ext)}")
        if plot_results:
            plt.show()

    def __connect_arduino(self, PORT):

        # Arduino pins
        PRESSURE_SENSOR_PIN = 3
        MOTOR_PWM_PIN = 5
        SOLENOID_PIN = 6

        board = pyfirmata.Arduino(PORT)
    
        board.analog[PRESSURE_SENSOR_PIN].enable_reporting()
        it = util.Iterator(board)
        it.start()
        time.sleep(0.05)
        self.pressure_sensor = board.get_pin(f'a:{PRESSURE_SENSOR_PIN}:i')
        self.motor_pwm = board.get_pin(f'd:{MOTOR_PWM_PIN}:p')

        mtr1_pin = board.get_pin('d:3:o')
        mtr2_pin = board.get_pin('d:4:o')
        mtr1_pin.write(True)
        mtr2_pin.write(False)
        
        self.solenoid = board.get_pin(f'd:{SOLENOID_PIN}:o')

        self.__arduino_connected = True
        
    def test_on_real_robot(self, P, I, D, SP, dt=0.1, total_run_time=10, PORT="COM4", save_to_file=False,save_image=False, plot_results=True):
        if not self.__arduino_connected:
            self.__connect_arduino(PORT)
        
        time_vector = []
        pressure_vector = []
        pressure_kalman_values = []

        self.solenoid.write(False) # close solenoid


        pid = PID(P, I, D, SP, dt, output_limits=(self.pwm_lower_limit, self.pwm_upper_limit))

        elapsed_time = 0
        while len(time_vector) < total_run_time / dt + 1:
            current_pressure = self.__arduino_get_pressure()

            if self.activate_kalman_filter:
                self.__kf.predict()
                self.__kf.update(current_pressure)
                current_pressure = self.__kf.x[0]

            output = round(pid(current_pressure))

            self.motor_pwm.write(output/255)
            print(f"Time: {elapsed_time:.4f} s, pressure: {current_pressure:.4f} kPa at PWM: {output}")
            time_vector.append(elapsed_time)
            pressure_vector.append(current_pressure)

            elapsed_time += dt

            time.sleep(dt)

        
        self.solenoid.write(True)
        self.motor_pwm.write(0)

        time_values = np.array(time_vector)
        pressure_values = np.array(pressure_vector)
        if self.activate_kalman_filter:
            pressure_kalman_values = pressure_values

        self.__print_metrics(SP, time_values, pressure_values, pressure_kalman_values, pid=(P, I, D), save_to_file=save_to_file, ga_params=None, real_gripper=True)
        self.__plot_results((P, I, D), SP, time_values, pressure_values, pressure_kalman_values,real_gripper=True, save_image=save_image, plot_results=plot_results)

    def __arduino_get_pressure(self) -> float:
        offset = 410
        fullScale = 9630

        pressure_analog_value = 0
        for i in range(10):
            pressure_analog_value += self.pressure_sensor.read()
        if isinstance(pressure_analog_value,numbers.Number):
            pressure_analog_value *= 1023
        else:
            raise Exception(f"pressure value is: {pressure_analog_value}")
        
        pressure = (pressure_analog_value - offset) * 700.0 / (fullScale - offset)
        
        return pressure

class GeneticPID:
    def __init__(self, soft_gripper_instance:SoftGripper, SP:float=40.0,
                 num_generations:int=300,
                 num_parents_mating=4,
                 sol_per_pop:int=80,
                 num_genes=3,
                 gene_space=[
                     {'low': 0, 'high': 20},  # Kp
                     {'low': 0, 'high': 8},  # Ki
                     {'low': 0, 'high': 1}   # Kd
                 ],
                 crossover_probability=0.1,
                 mutation_probability=0.5,
                 parent_selection_type="rank",
                 crossover_type="single_point",
                 mutation_type="random",
                 keep_parents=3
                ):
        self.num_generations = num_generations
        self.num_parents_mating=num_parents_mating
        self.sol_per_pop = sol_per_pop
        self.num_genes = num_genes
        self.gene_space = gene_space
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.parent_selection_type = parent_selection_type
        self.crossover_type= crossover_type
        self.mutation_type= mutation_type
        self.keep_parents = keep_parents
        
        self.__soft_gripper_instace=soft_gripper_instance
        self.SP = SP

        self.ga_instance = pygad.GA(
            num_generations=num_generations,
            num_parents_mating=num_parents_mating,
            fitness_func=self.fitness_function,
            sol_per_pop=sol_per_pop, 
            num_genes=num_genes,
            gene_space=gene_space,
            crossover_probability=crossover_probability,
            mutation_probability=mutation_probability,
            parent_selection_type=parent_selection_type,
            crossover_type=crossover_type,
            mutation_type=mutation_type,
            keep_parents=keep_parents

        )
        self.P = None
        self.I = None
        self.D = None
        self.best_fitness = None

    def fitness_function(self, ga_instance, solution, solution_idx):

        P, I, D = solution
        current_pressure = 0 
        pwm = 0 
        time = 0
        total_error = 0
        dt = self.__soft_gripper_instace.dt
        

        pid = PID(P, I, D, setpoint=self.SP, sample_time=None)
        pid.output_limits = (self.__soft_gripper_instace.pwm_lower_limit, self.__soft_gripper_instace.pwm_upper_limit)

        time_vector = []
        pressure_vector = []

        for _ in range(self.__soft_gripper_instace.num_steps):
            pwm = pid(current_pressure, dt=dt)
            pwm = round(pwm)
            
            current_pressure = self.__soft_gripper_instace.motor_pressure_response(pwm, time, current_pressure)
            
            time_vector.append(time)
            pressure_vector.append(current_pressure)

            time += dt

        time_vector = np.array(time_vector)
        pressure_vector = np.array(pressure_vector)


        IAE = self.__soft_gripper_instace.get_IAE(time_vector, pressure_vector, dt, self.SP)
        rise_time = self.__soft_gripper_instace.get_rise_time(time_vector, pressure_vector, self.SP)
        settling_time = self.__soft_gripper_instace.get_settling_time(time_vector, pressure_vector, dt, self.SP)
        overshoot = self.__soft_gripper_instace.get_overshoot(time_vector, pressure_vector, self.SP)
        steady_state_error = self.__soft_gripper_instace.get_steady_state_error(time_vector, pressure_vector, self.SP)

        if rise_time == None or settling_time == None:
            return -float("inf")

                
        total_error = w_iae * IAE + w_rs * rise_time\
            + w_st * settling_time + w_os * overshoot \
            + w_sse * steady_state_error

        return -total_error
        
    def run(self):
        self.ga_instance.run()
        best_solution, self.best_fitness, _ = self.ga_instance.best_solution()
        self.P, self.I, self.D = best_solution
        print(f"P, I, D = {self.P}, {self.I}, {self.D}")
        print(f"Best fitness:\n{w_iae}*IAE + \n{w_rs}*RS + \n{w_st}*ST + \n{w_os}*OS + \n{w_sse}*SSE \n-------\n{-self.best_fitness}")

    def plot_fitness(self):
        self.ga_instance.plot_fitness()

    def test_PID_values(self, save_to_file=False, plot_results=True, save_image=False):
        self.__soft_gripper_instace.pid_value_test(
            P=self.P,
            I=self.I,
            D=self.D,
            SP=self.SP,
            save_to_file=save_to_file,
            ga_params = (self.sol_per_pop, self.mutation_probability, self.crossover_probability, self.num_generations),
            plot_results=plot_results,
            save_image=save_image,
            best_fitness=-self.best_fitness)

class MotorPressureModelTest:
    def __init__(self, soft_gripper_instance:SoftGripper):
        self.__soft_gripper_instace=soft_gripper_instance
        self.__pwm_value = self.__soft_gripper_instace.pwm_lower_limit

        self.root = tk.Tk()
        self.root.title("PWM Control")

        self.__pwm_label = ttk.Label(self.root, text=f"PWM Value: {self.__pwm_value}")
        self.__pwm_label.pack(pady=10)

        increase_button = ttk.Button(self.root, text="Increase PWM", command=lambda: self.__update_pwm(5))
        increase_button.pack(pady=5)

        decrease_button = ttk.Button(self.root, text="Decrease PWM", command=lambda: self.__update_pwm(-5))
        decrease_button.pack(pady=5)

        self.__last_pressure = 0

    def __del__(self):
        self.root.destroy()
        self.plot_thread.join()
    
    def __update_pwm(self, step):
        toggle_motor_check_1 = self.__pwm_value == self.__soft_gripper_instace.pwm_lower_limit or self.__pwm_value == 0
        toggle_motor_check_2 = not (self.__pwm_value == self.__soft_gripper_instace.pwm_lower_limit and step > 0)
        
        if toggle_motor_check_1 and toggle_motor_check_2:
            if step < 0:
                self.__pwm_value = 0 
            else:
                self.__pwm_value =  self.__soft_gripper_instace.pwm_lower_limit 

        else:
            self.__pwm_value += step
            self.__pwm_value = max(self.__soft_gripper_instace.pwm_lower_limit, min(self.__soft_gripper_instace.pwm_upper_limit, self.__pwm_value))
        
        self.__pwm_label.config(text=f"PWM Value: {self.__pwm_value}")

    def run(self):
        self.plot_thread = threading.Thread(target=self.plot_pressure_response)
        self.plot_thread.start()
        self.root.mainloop()

    def plot_pressure_response(self):
        fig, ax = plt.subplots()
        x_data, y_data = [], []
        line, = ax.plot(x_data, y_data)
        
        y_max = 100
        ax.set_xlim(0, 10)
        ax.set_ylim(0, y_max)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Pressure')

        start_time = time.time()
        def update(frame):
            nonlocal x_data, y_data
            
            current_time = time.time() - start_time
            pressure = self.__soft_gripper_instace.motor_pressure_response(self.__pwm_value, current_time, self.__last_pressure)
            self.__last_pressure = pressure

            x_data.append(current_time)
            y_data.append(pressure)
            
            if len(x_data) > 100:
                x_data = x_data[-100:]
                y_data = y_data[-100:]
        

            line.set_xdata(x_data)
            line.set_ydata(y_data)
            
            ax.set_xlim(x_data[0], x_data[-1])
            ax.set_title(f"Pressure - Time at PWM: {self.__pwm_value}")
            
            return line,
        ani = FuncAnimation(fig, update, blit=False, interval=100)
        plt.show()

class SoftGripperKalmanFilter:
    def __init__(self):
        self.kf = KalmanFilter(dim_x=1, dim_z=1)
        self.kf.x = np.array([0]) # initial pressure
        self.kf.F = np.array([[1.0]]) # state transition matrix
        self.kf.H = np.array([[1.0]]) # measurement func
        self.kf.P = np.array([[1.0]]) # covariance matrix / initial uncertainty
        self.kf.R = np.array([[0.5]]) # measurement uncertainty
        self.kf.Q = np.array([[0.05]]) # process uncertainty