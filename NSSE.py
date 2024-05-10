import numpy as np
import math

# ハイパーパラメータ
WPOP_SIZE = 400         # 全体解集団のサイズ
PPOP_SIZE = 400         # 部分解集団のサイズ
MAX_GENERATION = 250    # 世代交代数
WCROSSOVER_PROB = 0.5   # 全体解集団の交叉率
PCROSSOVER_PROB = 0.5   # 部分解集団の交叉率
WMUTATE_PROB = 0.05     # 全体解遺伝子の突然変異確率
PMUTATE_PROB = 0.2      # 部分解遺伝子の突然変異確率
WCHROM_LEN = 100        # 全体解個体のサイズ
PCHROM_LEN = 20         # 部分解集団のサイズ
TOURNAMENT_SIZE = 20    # トーナメントサイズ

# 部分解個体
class PartialIndividual:
    def __init__(self):
        self.chrom = np.random.randint(0, 2, PCHROM_LEN)
        self.global_fitness = float('inf')

    def crossover(self, parent1, parent2, index1, index2):
        if index1 > index2:
            tmp = index1
            index1 = index2
            index2 = tmp
        for i in range(0, index1):
            self.chrom[i] = parent1.chrom[i]
        for i in range(index1, index2):
            self.chrom[i] = parent2.chrom[i]
        for i in range(index2, PCHROM_LEN):
            self.chrom[i] = parent1.chrom[i]
        self.mutate()
    
    def mutate(self):
        for i in range(PCHROM_LEN):
            if np.random.rand() < PMUTATE_PROB:
                self.chrom[i] = 1 - self.chrom[i]

# 部分解集団
class PartialPopulation:
    def __init__(self):
        self.population = []
        for i in range(PPOP_SIZE):
            individual = PartialIndividual()
            self.population.append(individual)
    
    def crossover(self):
        for i in range(int(PPOP_SIZE * (1 - PCROSSOVER_PROB)), PPOP_SIZE):
            # 二点交叉
            parent1 = min(np.random.choice(range(PPOP_SIZE), TOURNAMENT_SIZE), key=lambda x: self.population[x].global_fitness)
            parent2 = min(np.random.choice(range(PPOP_SIZE), TOURNAMENT_SIZE), key=lambda x: self.population[x].global_fitness)
            index1 = np.random.randint(0, PCHROM_LEN)
            index2 = np.random.randint(0, PCHROM_LEN)
            self.population[i].crossover(self.population[parent1], self.population[parent2], index1, index2)

    def evainit(self):
        for i in range(PPOP_SIZE):
            self.population[i].global_fitness = float('inf')


# 全体解個体
class WholeIndividual:
    def __init__(self, ppop):
        self.chrom = []
        self.ppop = ppop
        for _ in range(WCHROM_LEN):
            index = np.random.randint(0, PPOP_SIZE)
            self.chrom.append(self.ppop.population[index])
        self.global_fitness = float('inf')
        self.rankfit = float('inf')
        self.cd = 0
        self.fitness1 = float('inf')
        self.fitness2 = float('inf')
    
    def crossover(self, parent1, parent2, index1, index2):
        if index1 > index2:
            tmp = index1
            index1 = index2
            index2 = tmp
        for i in range(0, index1):
            self.chrom[i] = parent1.chrom[i]
        for i in range(index1, index2):
            self.chrom[i] = parent2.chrom[i]
        for i in range(index2, WCHROM_LEN):
            self.chrom[i] = parent1.chrom[i]
        self.mutate()
    
    def mutate(self):
        for i in range(WCHROM_LEN):
            if np.random.rand() < WMUTATE_PROB:
                index = np.random.randint(0, PPOP_SIZE)
                self.chrom[i] = self.ppop.population[index]

# 全体解集団
class WholePopulation:
    def __init__(self, ppop):
        self.population = []
        for i in range(WPOP_SIZE):
            individual = WholeIndividual(ppop)
            self.population.append(individual)
    
    def crossover(self):
        for i in range(int(WPOP_SIZE * (1 - WCROSSOVER_PROB)), WPOP_SIZE):
            # 二点交叉
            parent1 = min(np.random.choice(range(WPOP_SIZE), TOURNAMENT_SIZE), key=lambda x: self.population[x].global_fitness)
            parent2 = min(np.random.choice(range(WPOP_SIZE), TOURNAMENT_SIZE), key=lambda x: self.population[x].global_fitness)
            index1 = np.random.randint(0, WCHROM_LEN)
            index2 = np.random.randint(0, WCHROM_LEN)
            self.population[i].crossover(self.population[parent1], self.population[parent2], index1, index2)

    def evainit(self):
        for i in range(int(WPOP_SIZE * (1 - WCROSSOVER_PROB)), WPOP_SIZE):
            self.population[i].global_fitness = float('inf')
            self.population[i].fitness1 = float('inf')
            self.population[i].fitness2 = float('inf')

# 各目的関数値の算出
def evaluate_object(wpop):
    for j in range(0, WPOP_SIZE):
        def gray_to_decimal(gray):
            binary_code = [0] * 20
            binary_code[0] = gray.chrom[0]
            for j in range(1, 20):
                binary_code[j] = binary_code[j - 1] ^ gray.chrom[j]
            decimal_value = 0
            for j in range(20):
                decimal_value += binary_code[j] * (2 ** (19 - j))
            return decimal_value

        def map_to_real(decimal_value, binary_max):
            return -5 + decimal_value * (10 / (binary_max - 0))

        binary_max = (2 ** 20) - 1
        real_value = map_to_real(gray_to_decimal(wpop.population[j].chrom[0]), binary_max)
        wpop.population[j].fitness1 = 0
        wpop.population[j].fitness2 = 0

        for i in range(1, 101):
            if i != 100:
                real_value2 = map_to_real(gray_to_decimal(wpop.population[j].chrom[i]), binary_max)
                wpop.population[j].fitness1 += -10 * math.exp(-0.2 * math.sqrt(real_value ** 2 + real_value2 ** 2))
                real_value = real_value2
            
            wpop.population[j].fitness2 += math.pow(abs(real_value), 0.8) + 5 * math.sin(real_value ** 3)

# 混雑距離
def crowding_distance(tmp_rank, wpop):
    for i in range(len(tmp_rank)):
        wpop.population[tmp_rank[i]].cd = 0

    if(len(tmp_rank) >= 2):
        tmp_rank = sorted(tmp_rank, key=lambda tmp_rank: wpop.population[tmp_rank].fitness1)
        wpop.population[tmp_rank[0]].cd = float('inf')
        wpop.population[len(tmp_rank) - 1].cd = float('inf')
        for i in range(1, len(tmp_rank) - 1):
            if(wpop.population[tmp_rank[len(tmp_rank) - 1]].fitness1 - wpop.population[tmp_rank[0]].fitness1 != 0):
                wpop.population[tmp_rank[i]].cd += (wpop.population[tmp_rank[i+1]].fitness1 - wpop.population[tmp_rank[i-1]].fitness1) / (wpop.population[tmp_rank[len(tmp_rank) - 1]].fitness1 - wpop.population[tmp_rank[0]].fitness1)

        tmp_rank = sorted(tmp_rank, key=lambda tmp_rank: wpop.population[tmp_rank].fitness2)
        wpop.population[tmp_rank[0]].cd = float('inf')
        wpop.population[len(tmp_rank) - 1].cd = float('inf')
        for i in range(1, len(tmp_rank) - 1):
            if(wpop.population[tmp_rank[len(tmp_rank) - 1]].fitness2 - wpop.population[tmp_rank[0]].fitness2 != 0):
                wpop.population[tmp_rank[i]].cd += (wpop.population[tmp_rank[i+1]].fitness2 - wpop.population[tmp_rank[i-1]].fitness2) / (wpop.population[tmp_rank[len(tmp_rank) - 1]].fitness2 - wpop.population[tmp_rank[0]].fitness2)

# 評価関数
def evaluate_fitness(wpop, ppop):
    eva_ind_cnt = 0
    rank = 1
    next_remain = []
    for i in range(WPOP_SIZE):
        next_remain.append(i)
    
    evaluate_object(wpop)
    
    while(eva_ind_cnt < WPOP_SIZE):
        tmp_eva_ind_cnt = eva_ind_cnt
        current_remain = next_remain
        next_remain = []
        tmp_rank = []

        for i in range(WPOP_SIZE - tmp_eva_ind_cnt):
            for j in range(WPOP_SIZE - tmp_eva_ind_cnt):
                flag = 1
                if(wpop.population[current_remain[i]].fitness1 >= wpop.population[current_remain[j]].fitness1
                and wpop.population[current_remain[i]].fitness2 >= wpop.population[current_remain[j]].fitness2
                and (wpop.population[current_remain[i]].fitness1 != wpop.population[current_remain[j]].fitness1 or wpop.population[current_remain[i]].fitness2 != wpop.population[current_remain[j]].fitness2)):
                    next_remain.append(current_remain[i])
                    flag = 0
                    break
            if(flag == 1):
                wpop.population[current_remain[i]].rankfit = rank
                tmp_rank.append(current_remain[i])
                eva_ind_cnt += 1

        crowding_distance(tmp_rank, wpop)
        rank += 1

    for i in range(WPOP_SIZE):
        wpop.population[i].global_fitness = wpop.population[i].rankfit + 1 / (wpop.population[i].cd * 100 + 1)
        for j in range(WCHROM_LEN):
            if(wpop.population[i].chrom[j].global_fitness > wpop.population[i].global_fitness):
                wpop.population[i].chrom[j].global_fitness = wpop.population[i].global_fitness
    wpop.population.sort(key=lambda individual: individual.global_fitness)
    ppop.population.sort(key=lambda individual: individual.global_fitness)
