import json
import numpy as np
import os


def generate_intermediate_vectors(v1, v2, n):
    intermediate_vectors = []
    for i in range(1, n + 1):
        intermediate_vector = v1 + (v2 - v1) * (i / (n + 1))
        intermediate_vectors.append(intermediate_vector)
    return intermediate_vectors

def main():
    data_files = os.listdir("./data")

    for file_name in data_files:
        if not file_name.startswith("pwm_pressure_data_") or file_name == "pwm_pressure_data_255.txt":
            continue
        pwm_val = int(file_name.split("_")[3].split(".")[0])
        
        if pwm_val == 250:
            next_pwm_val = pwm_val + 5
            number_of_intermediate_vectors = 4
        else:
            next_pwm_val = pwm_val + 10
            number_of_intermediate_vectors = 9

        lower_file_name = os.path.join(".", "data", file_name)
        upper_file_name = os.path.join(".", "data", file_name.replace(str(pwm_val), str(next_pwm_val)))

        lower_file = open(lower_file_name)
        upper_file = open(upper_file_name)
        lower_data = json.load(lower_file)
        upper_data = json.load(upper_file)
        lower_file.close()
        upper_file.close()

        lower_pressure_vector = np.array(list(lower_data[str(pwm_val)].values()), dtype=np.float32)
        lower_time_vector = np.array(list(lower_data[str(pwm_val)].keys()), dtype=np.float32)
        upper_pressure_vector = np.array(list(upper_data[str(next_pwm_val)].values()), dtype=np.float32)
        upper_time_vector = np.array(list(upper_data[str(next_pwm_val)].keys()), dtype=np.float32)

        min_vector_len = min(len(upper_time_vector), len(lower_time_vector))
        
        lower_pressure_vector = lower_pressure_vector[:min_vector_len]
        upper_pressure_vector = upper_pressure_vector[:min_vector_len]

        lower_time_vector = lower_time_vector[:min_vector_len]
        upper_time_vector = upper_time_vector[:min_vector_len]

        assert np.equal(lower_time_vector, upper_time_vector).all()

        intermediate_pressure_vectors = generate_intermediate_vectors(lower_pressure_vector, upper_pressure_vector, number_of_intermediate_vectors)

        for increase_pwm, pressure_vector in enumerate(intermediate_pressure_vectors, start=1):
            current_pwm = pwm_val + increase_pwm

            data_dict = {}
            data_dict[str(current_pwm)] = {str(lower_time_vector[i]): float(pressure_vector[i]) for i in range(min_vector_len)}

            new_file_name = file_name.replace(str(pwm_val), str(current_pwm))

            new_file_path = os.path.join(".", "data", new_file_name)
            new_file = open(new_file_path, "w")
            new_file.write(json.dumps(data_dict, indent=4))
            new_file.close()

if __name__ == "__main__":
    main()
