import numpy as np
import random
from typing import List, Tuple, Optional
class GeneticAlgorithm:
    def __init__(
        self,
        pop_size: int,      # Population size
        generations: int,   # Number of generations for the algorithm
        mutation_rate: float,  # Gene mutation rate
        crossover_rate: float,  # Gene crossover rate
        tournament_size: int,  # Tournament size for selection
        elitism: bool,         # Whether to apply elitism strategy
        random_seed: Optional[int],  # Random seed for reproducibility
    ):
        # Students need to set the algorithm parameters
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.elitism = elitism

        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

    def _init_population(self, M: int, N: int) -> List[List[int]]:
        """
        Initialize the population and generate random individuals, ensuring that every student is assigned at least one task.
        :param M: Number of students
        :param N: Number of tasks
        :return: Initialized population
        """
        # TODO: Initialize individuals based on the number of students M and number of tasks N
        Ans =[] #return list
        for i in range(self.pop_size):
            init = [i for i in range(M)]
            init=init +random.sample(range(M),N-M)
            random.shuffle(init)
            Ans.append(init)
        return Ans    
    def _fitness(self, individual: List[int], student_times: np.ndarray) -> float:
        """
        Fitness function: calculate the fitness value of an individual.
        :param individual: Individual
        :param student_times: Time required for each student to complete each task
        :return: Fitness value
        """
        # TODO: Design a fitness function to compute the fitness value of the allocation plan
        sc = 0 # toatal score 
        for i in range(len(individual)):
            sc += student_times[individual[i]][i]
        return sc

    def _selection(self, population: List[List[int]], fitness_scores: List[float]) -> List[int]:#想一下要不要限制不要重複取
        """
        Use tournament selection to choose parents for crossover.
        :param population: Current population
        :param fitness_scores: Fitness scores for each individual
        :return: Selected parent
        """
        # TODO: Use tournament selection to choose parents based on fitness scores
        
        random_select = random.sample(range(len(population)),self.tournament_size) 
        #from population random select  tournament_size people
        win = random_select[0] #to record best index
        for select in random_select:
            if fitness_scores[select] < fitness_scores[win]:
                win = select
        return population[win]

    def _crossover(self, parent1: List[int], parent2: List[int], M: int) -> Tuple[List[int], List[int]]:
        """
        Crossover: generate two offspring from two parents.
        :param parent1: Parent 1
        :param parent2: Parent 2
        :param M: Number of students
        :return: Generated offspring
        # """
        child1 = parent1
        child2 = parent2
        if random.random() <= self.crossover_rate:
            cut = random.randint(1,len(parent1)-1) # select a point to cut
            if cut == len(parent1)-1 :
                return (parent1 , parent2)
            else:
                legel = False
                while not legel:
                    cut = random.randint(1,len(parent1)-1)
                    if cut ==len(parent1)-1:
                        break
                    else:
                        child1 = parent1[:cut] + parent2[cut:]
                        child2 = parent2[:cut] + parent1[cut:]
                        for i in range(M):
                            if (i in child1) and (i in child2):
                                legel = True
                            else:
                                legel = False
                                child1 = parent1
                                child2 = parent2
                return (child1, child2)
        else:
            return (parent1 , parent2)
                
    def _mutate(self, individual: List[int], M: int) -> List[int]:
        """
        Mutation operation: randomly change some genes (task assignments) of the individual.
        :param individual: Individual
        :param M: Number of students
        :return: Mutated individual
        """
        legal =False
        aug = individual
        while not legal:
            chage = random.randint(0,M-1)
            task = random.randint(0,N-1) 
            original = individual[task]
            individual[task] = chage
            for i in range(M):
                if i in individual:
                    legal = True
                else:
                    legal =False
                    individual[task] = original
                    break
        if self._fitness(aug,student_times) < self._fitness(individual,student_times):
            return aug
        else:
            return individual

    def __call__(self, M: int, N: int, student_times: np.ndarray) -> Tuple[List[int], int]:
        """
        Execute the genetic algorithm and return the optimal solution (allocation plan) and its total time cost.
        :param M: Number of students
        :param N: Number of tasks
        :param student_times: Time required for each student to complete each task
        :return: Optimal allocation plan and total time cost
        """
        # TODO: Complete the genetic algorithm process, including initialization, selection, crossover, mutation, and elitism strategy
        init_set = self._init_population(M,N)
        init_score_set = [self._fitness(individual,student_times) for individual in init_set]
        parent_set = []
        parent_set_score = []
        for i in range(self.pop_size):
            parent_set.append(self._selection(init_set,init_score_set))
        for i in parent_set:
            parent_set_score.append(self._fitness(i,student_times))
        best = 0
        for i in range(self.pop_size):
            if parent_set_score[i] <parent_set_score[best]:
                best = i
        best_set = parent_set[best]
        best_score = parent_set_score[best]
        for g in range(self.generations):
            child_set = []
            child_score_set = []
            
            for i in range(0,self.pop_size,2):
                cross = []
                if i+1 == self.pop_size:
                    cross = list(self._crossover(parent_set[i],parent_set[0],M))
                else:
                    cross = list(self._crossover(parent_set[i],parent_set[i+1],M))
                child_set.append(cross[0])
                child_set.append(cross[1])
            
            for i in range(len(child_set)):
                child_set[i]=self._mutate(child_set[i],M)
                child_score_set.append(self._fitness(child_set[i],student_times))
            
            for i in range(len(child_set)):
                if child_score_set[i] <best_score:
                    best_score = child_score_set[i]
                    best_set = []
                    best_set.append(child_set[i])
            parent_set = []
            for i in range(self.pop_size):
                parent_set.append(self._selection(child_set,child_score_set))
        return (list(best_set) ,int(best_score))

if __name__ == "__main__":
    def write_output_to_file(problem_num: int, total_time: int, filename: str = "answer.txt") -> None:
        """
        Write results to a file and check if the format is correct.
        """
        print(f"Problem {problem_num}: Total time = {total_time}")

        if not isinstance(total_time, int) :
            raise ValueError(f"Invalid format for problem {problem_num}. Total time should be an integer.")
        
        with open(filename, 'a') as file:
            file.write(f"Total time = {total_time}\n")

    # TODO: Define multiple test problems based on the examples and solve them using the genetic algorithm
    # Example problem 1 (define multiple problems based on the given example format)
    M1, N1 = 2, 3
    cost1 = [[3,2,4],
             [4,3,2]]
    
    M2, N2 = 4, 4
    cost2 = [[5,6,7,4],
             [4,5,6,3],
             [6,4,5,2],
             [3,2,4,5]]
    
    M3, N3 = 8, 9
    cost3 = [[90, 100, 60, 5, 50, 1, 100, 80, 70],
             [100, 5, 90, 100, 50, 70, 60, 90, 100],
             [50, 1, 100, 70, 90, 60, 80, 100, 4],
             [60, 100, 1, 80, 70, 90, 100, 50, 100],
             [70, 90, 50, 100, 100, 4, 1, 60, 80],
             [100, 60, 100, 90, 80, 5, 70, 100, 50],
             [100, 4, 80, 100, 90, 70, 50, 1, 60],
             [1, 90, 100, 50, 60, 80, 100, 70, 5]]
        
    M4, N4 = 3, 3
    cost4 =[[2,5,6],
            [4,3,5],
            [5,6,2]] 
    
    M5, N5 = 4, 4
    cost5 =[[4,5,1,6],
            [9,1,2,6],
            [6,9,3,5],
            [2,4,5,2]]

    M6, N6 = 4, 4
    cost6 = [[5,4,6,7],
             [8,3,4,6],
             [6,7,3,8],
             [7,8,9,2]]
    
    M7, N7 = 4, 4
    cost7 = [[4, 7, 8, 9],
             [6, 3, 6, 7],
             [8, 6, 2, 6],
             [7, 8, 7, 3]]
    
    M8, N8 = 5 , 5
    cost8 = [[8, 8, 24, 24, 24],
             [6, 18, 6, 18, 18],
             [30, 10, 30, 10, 30],
             [21, 21, 21, 7, 7],
             [27, 27, 9, 27, 9]]
    
    M9, N9 = 5,5
    cost9 = [[10, 10, 99, 99, 99],
             [12, 99, 99, 99, 12],
             [99, 15, 15, 99, 99],
             [11, 99, 99, 99, 99],
             [99, 14, 99, 14, 99]]
    
    M10, N10 = 9 , 10 
    cost10 = [[1, 90, 100, 50, 70, 20, 100, 60, 80, 90],
              [100, 10, 1, 100, 60, 80, 70, 100, 50, 90],
              [90, 50, 70, 1, 100, 100, 60, 90, 80, 100],
              [70, 100, 90, 5, 10, 60, 100, 80, 90, 50],
              [50, 100, 100, 90, 20, 4, 80, 70, 60, 100],
              [100, 5, 80, 70, 90, 100, 4, 50, 1, 60],
              [90, 60, 50, 4, 100, 90, 100, 5, 10, 80],
              [100, 70, 90, 100, 4, 60, 1, 90, 100, 5],
              [80, 100, 5, 60, 50, 90, 70, 100, 4, 1]]
    
    problems = [(M1, N1, np.array(cost1)),
                (M2, N2, np.array(cost2)),
                (M3, N3, np.array(cost3)),
                (M4, N4, np.array(cost4)),
                (M5, N5, np.array(cost5)),
                (M6, N6, np.array(cost6)),
                (M7, N7, np.array(cost7)),
                (M8, N8, np.array(cost8)),
                (M9, N9, np.array(cost9)),
                (M10, N10, np.array(cost10))]
    # Example for GA execution:
    # TODO: Please set the parameters for the genetic algorithm
    ga = GeneticAlgorithm(
        pop_size = 200,
        generations=150,
        mutation_rate=0.1,
        crossover_rate=0.7,
        tournament_size=20,
        elitism=True,
        random_seed=124
    )
    # Solve each problem and immediately write the results to the file
    for i, (M, N, student_times) in enumerate(problems, 1):
            best_allocation, total_time = ga(M=M, N=N, student_times=student_times)
            write_output_to_file(i, total_time)
    print("Results have been written to results.txt")
