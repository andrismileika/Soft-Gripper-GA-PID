from Soft_gripper_PID import *
import time

TEST = 1
TOTAL_TEST_TIME = 10

start_time = time.time()
X = SoftGripper(dt=0.05, sim_time=TOTAL_TEST_TIME, kalman_filter=True)


if TEST == 1:

    geneticPID = GeneticPID(
        soft_gripper_instance=X,
        SP=30.0,
        num_generations=300,
        num_parents_mating=4,
        sol_per_pop=100,
        num_genes=3,
        gene_space=[
            {'low': 0, 'high': 20},  # Kp
            {'low': 0, 'high': 8},  # Ki
            {'low': 0, 'high': 1}   # Kd
        ],
        crossover_probability=0.8,
        mutation_probability=0.1,
        parent_selection_type="rank",
        crossover_type="single_point",
        mutation_type="random",
    )

    geneticPID.run()
    geneticPID.plot_fitness()
    geneticPID.test_PID_values(save_to_file=True)#, plot_results=False)
    X.reset_kalman_filter()

elif TEST == 3: # test gripper
    P, I, D =16.84044406,	2.939955986,	0.012374878

    SP =40
    X.test_on_real_robot(P, I, D, SP, dt=0.05, total_run_time=TOTAL_TEST_TIME, PORT="COM3", save_to_file=True, save_image=True, plot_results=True)

elif TEST == 5:
    # loop through simulation
    SET_POINTS = [30, 40, 50]
    POPULATION_SIZES = [20, 80, 150]
    MUTATION_RATES = [0.01, 0.5, 1]
    CROSSOVER_RATES = [0.01, 0.5, 1]
    NUMBER_OF_GENERATIONS = [100, 200, 400]

    FIXED_POP_SIZE = 100
    FIXED_MUT_RATE = 0.1
    FIXED_CROSS_RATE = 0.8
    FIXED_NUM_GEN = 300
    
    VARIABLES = ["POPULATION_SIZE", "MUTATION_RATE", "CROSSOVER_RATE", "NUM_GENERATIONS"]

    for sp in SET_POINTS:
        print(f"Performing simulations with SP={sp}")
        
        for variable in VARIABLES:
            print(f"Current variable: {variable}")

            if variable == "POPULATION_SIZE":
                for pop_size in POPULATION_SIZES:
                    for i in range(3):
                        print(f"Performing simulation with variable {variable}")
                        print(f"Population size = {pop_size}")
                        print(f"Mutation rate = {FIXED_MUT_RATE}\nCrossover rate = {FIXED_CROSS_RATE}\nNumber of generations = {FIXED_NUM_GEN}")
                        
                        geneticPID = GeneticPID(
                            soft_gripper_instance=X,
                            SP=sp,
                            num_generations=FIXED_NUM_GEN,
                            crossover_probability=FIXED_CROSS_RATE,
                            mutation_probability=FIXED_MUT_RATE,
                            sol_per_pop=pop_size
                        )
                        try:
                            geneticPID.run()
                            geneticPID.test_PID_values(save_to_file=True, plot_results=False, save_image=True)
                        except:
                            print("Simulation failed. Skipping")
                            continue
                        X.reset_kalman_filter()
                        print("==========================================")
            elif variable == "MUTATION_RATE":
                for mr in MUTATION_RATES:
                    for i in range(3):
                        print(f"Performing simulation with variable {variable}")
                        print(f"Mutation rate = {mr}")
                        print(f"Population size = {FIXED_POP_SIZE}\nCrossover rate = {FIXED_CROSS_RATE}\nNumber of generations = {FIXED_NUM_GEN}")
                        
                        geneticPID = GeneticPID(
                            soft_gripper_instance=X,
                            SP=sp,
                            num_generations=FIXED_NUM_GEN,
                            crossover_probability=FIXED_CROSS_RATE,
                            mutation_probability=mr,
                            sol_per_pop=FIXED_POP_SIZE
                        )
                        try:
                            geneticPID.run()
                            geneticPID.test_PID_values(save_to_file=True, plot_results=False, save_image=True)
                        except:
                            print("Simulation failed. Skipping")
                            continue
                        X.reset_kalman_filter()
                        print("==========================================")
            
            elif variable == "CROSSOVER_RATE":
                for cr in CROSSOVER_RATES:
                    for i in range(3):
                        print(f"Performing simulation with variable {variable}")
                        print(f"Crossover rate = {cr}")
                        print(f"Population size = {FIXED_POP_SIZE}\nMutation rate = {FIXED_MUT_RATE}\nNumber of generations = {FIXED_NUM_GEN}")
                        
                        geneticPID = GeneticPID(
                            soft_gripper_instance=X,
                            SP=sp,
                            num_generations=FIXED_NUM_GEN,
                            crossover_probability=cr,
                            mutation_probability=FIXED_MUT_RATE,
                            sol_per_pop=FIXED_POP_SIZE
                        )
                        try:
                            geneticPID.run()
                            geneticPID.test_PID_values(save_to_file=True, plot_results=False, save_image=True)
                        except:
                            print("Simulation failed. Skipping")
                            continue
                        X.reset_kalman_filter()
                        print("==========================================")
            
            elif variable == "NUM_GENERATIONS":
                for num_gen in NUMBER_OF_GENERATIONS:
                    for i in range(3):
                        print(f"Performing simulation with variable {variable}")
                        print(f"Number of generations = {num_gen}")
                        print(f"Population size = {FIXED_POP_SIZE}\nCrossover rate = {FIXED_CROSS_RATE}\nMutation rate = {FIXED_MUT_RATE}")
                        
                        geneticPID = GeneticPID(
                            soft_gripper_instance=X,
                            SP=sp,
                            num_generations=num_gen,
                            crossover_probability=FIXED_CROSS_RATE,
                            mutation_probability=FIXED_MUT_RATE,
                            sol_per_pop=FIXED_POP_SIZE
                        )
                        try:
                            geneticPID.run()
                            geneticPID.test_PID_values(save_to_file=True, plot_results=False, save_image=True)
                        except:
                            print("Simulation failed. Skipping")
                            continue
                        X.reset_kalman_filter()
                        print("==========================================")

print(f"Total time taken: {time.time()-start_time} seconds")
