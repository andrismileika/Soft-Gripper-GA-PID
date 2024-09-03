import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os


def hyperbolic_func(x, a, b, c):
    return a / (-x - b) + c

def main(plot=False):


    data_files = os.listdir("./data")
    for filename in data_files:
        if not filename.startswith("pwm_pressure_data_"):
            continue
        
        pwm = filename.split("_")[3].split(".")[0]

        file_path = os.path.join(".", "data", filename)
        in_file = open(file_path,'r')
        data = json.load(in_file)
        in_file.close()

        new_data = {}

        pressure_vector = np.array(list(data[str(pwm)].values()), dtype=np.float32)
        time_vector = np.array(list(data[str(pwm)].keys()), dtype=np.float32)

        initial_guess = [1, 1, np.max(pressure_vector)]

        success = False
        n= 0 
        while not success:
            try:
                params, covariance = curve_fit(hyperbolic_func, time_vector[:-n], pressure_vector[:-n], p0=initial_guess)
                success = True
            except:
                n += 1
                if n > 50:
                    raise Exception(f"Failed at pwm {pwm}. Reached {n} try limit")


        a, b, c = params
        new_data[pwm] = {}
        new_data[pwm]['a'] = a
        new_data[pwm]['b'] = b
        new_data[pwm]['c'] = c

        if plot:
            x_new = np.arange(0, np.max(time_vector)*1.5, 0.05)
            y_fitted = hyperbolic_func(x_new, *params)
            plt.plot(time_vector, pressure_vector, 'o', x_new, y_fitted, '-')
            plt.show()
            
        out_file_name = os.path.join(".", "data", f"pwm_hyperbolic_coefficients_{pwm}.txt")
        out_file = open(out_file_name, 'w')
        out_file.write(json.dumps(new_data, indent=4))
        print(f"Saving coefficients to {out_file_name}")
        out_file.close()

if __name__ == "__main__":
    plot = False 
    main(plot)