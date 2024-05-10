import NSSE as se
# 初期化
ppop = se.PartialPopulation()
wpop = se.WholePopulation(ppop)
se.evaluate_fitness(wpop, ppop)

# 世代交代
for i in range(se.MAX_GENERATION):
    print(f"第{i+1}世代")
    # 交叉
    ppop.crossover()
    wpop.crossover()

    # 適応度初期化
    ppop.evainit()
    wpop.evainit()

    # 適応度算出
    se.evaluate_fitness(wpop, ppop)

for i in range(int(se.WPOP_SIZE / 2)):
    print(f"{wpop.population[i].fitness1}, {wpop.population[i].fitness2}")
    # print(f"{wpop.population[i].cd}")