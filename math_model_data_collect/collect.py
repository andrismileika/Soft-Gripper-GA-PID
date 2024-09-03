import time
import pyfirmata
import pyfirmata.util as util
import json
import random
import traceback

PWM_VALUES = [255]#(80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 255)

SAMPLING_TIME = 8
D_T = 0.05
PORT = "COM3"
PRESSURE_SENSOR_PIN = 3
MOTOR_PWM_PIN = 5
SOLENOID_PIN = 6

data = {}

def get_pressure(pressure_sensor: pyfirmata.Pin) -> float:
    offset = 410
    fullScale = 9630

    pressure_analog_value = 0
    for i in range(10):
        pressure_analog_value += pressure_sensor.read() 
    if pressure_analog_value:
        pressure_analog_value *= 1023
    else:
        raise Exception(f"pressure value is: {pressure_analog_value}")
    
    pressure = (pressure_analog_value - offset) * 700.0 / (fullScale - offset)
    
    return pressure


def main():
    board = pyfirmata.Arduino(PORT)
    
    board.analog[PRESSURE_SENSOR_PIN].enable_reporting()
    it = util.Iterator(board)
    it.start()
    time.sleep(0.05)
    pressure_sensor = board.get_pin(f'a:{PRESSURE_SENSOR_PIN}:i')
    motor_pwm = board.get_pin(f'd:{MOTOR_PWM_PIN}:p')

    mtr1_pin = board.get_pin('d:3:o')
    mtr2_pin = board.get_pin('d:4:o')
    mtr1_pin.write(True)
    mtr2_pin.write(False)
    
    solenoid = board.get_pin(f'd:{SOLENOID_PIN}:o')

    filename = f"pwm_pressure_data_{random.randint(0,1000)}.txt"
    file = open(filename, "w")

    for pwm in PWM_VALUES:
        try:
            print(f"Trying PWM: {pwm}")
            data[pwm] = {}
            solenoid.write(False)
            motor_pwm.write(pwm/255)
            
            t_start = time.time()
            t = 0
            while (time.time() - t_start < SAMPLING_TIME):
                pressure_pin_reading = get_pressure(pressure_sensor)
                data[pwm][t] = pressure_pin_reading
                t = round(t+D_T,2)
                print(f"{t} : {pressure_pin_reading}")
                time.sleep(D_T)

            solenoid.write(True)
            motor_pwm.write(0)
            time.sleep(2)
        except Exception:
            print(f"FAILED AT PWM: {pwm}")
            traceback.print_exc()
            file.write(json.dumps(data, indent=4))
            print(f"Saving to {filename}")
            file.close()
            exit()

    print("Success!")   
    file.write(json.dumps(data, indent=4))
    print(f"Saving to {filename}")
    file.close()
        


if __name__ == "__main__":
    # main program
    main()



