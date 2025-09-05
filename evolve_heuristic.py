import argparse
import os
import re
from src.pipeline.heuristic_evolver import HeuristicEvolver
from src.pipeline.ME_evolver import MapElitesEvolver
from src.util.util import search_file
from src.util.llm_client.get_llm_client import get_llm_client


def parse_arguments():
    problem_pool = [problem for problem in os.listdir(os.path.join("src", "problems")) if problem != "base"]

    parser = argparse.ArgumentParser(description="Evolve heuristic")
    parser.add_argument("-p", "--problem", choices=problem_pool, required=True, help="Specifies the type of combinatorial optimization problem.")
    parser.add_argument("-m", "--smoke_test", action='store_true', help="Optional flag to conduct a preliminary smoke test.")
    parser.add_argument("-l", "--llm_config_file", type=str, default=os.path.join("output", "llm_config", "azure_gpt_4o.json"), help="Path to the language model configuration file. Default is azure_gpt_4o.json.")
    parser.add_argument("-e", "--seed_heuristic", type=str, required=True, help="The initial seed heuristic to be evolved.")
    parser.add_argument("-ed", "--evolution_data", type=str, default="evolution_data", help="Directory containing the training dataset used for heuristic evolution.")
    parser.add_argument("-vd", "--validation_data", type=str, default=None, help="Directory containing the validation dataset used to evaluate heuristic performance.")
    parser.add_argument("-pe", "--perturbation_heuristic", type=str, default=None, help="Optional name or path of an additional perturbation heuristic to explore more diverse strategies. Default is random_xxxx.py in src/problems/{problem}/heuristic/basic_heuristic")
    parser.add_argument("-pr", "--perturbation_ratio", type=float, default=0.1, help="Proportion of operations to be randomly altered during each perturbation cycle. Default is 0.1.")
    parser.add_argument("-pt", "--perturbation_time", type=int, default=1000, help="Maximum number of perturbation cycles performed per evolution round.Default is 1000.")
    parser.add_argument("-i", "--max_refinement_round", type=int, default=3, help="Total number of refinement rounds to iteratively improve heuristics. Default is 5.")
    parser.add_argument("-f", "--filter_num", type=int, default=1, help="Number of top-performing heuristics to retain after each validation phase. Default is 3.")
    parser.add_argument("-r", "--evolution_rounds", type=int, default=3, help="Total number of evolutionary iterations to perform. Default is 3.")

    
    # MAP-Elites specific
    parser.add_argument("--algo", choices=["evolution", "map-elites"], default="evolution", help="Choose baseline evolution or MAP-Elites.")
    parser.add_argument("--me-bins", type=str, default="5,5,5", help="Bins per BD dimension, e.g. '10,10,10'.")
    parser.add_argument("--me-bd", type=str, default="SLOC, halstead, cyclomatic_complexity", help="Behavior descriptors used")
    parser.add_argument("--me-range", type=str, default="0:300,0:200,0:50", help="BD ranges 'min:max,min:max' for discretization.")
    parser.add_argument("--me-init", type=int, default=8, help="Number of initial offspring to generate from the seed.")
    parser.add_argument("--me-iterations", type=int, default=5, help="Number of iterations for the MAP-Elites algorithm.")
    parser.add_argument("--me-batch", type=int, default=8, help="Number of parents to sample and evolve per iteration.")
    return parser.parse_args()

def main():
    args = parse_arguments()
    problem = args.problem
    basic_heuristic_file = args.seed_heuristic
    evolution_dir = args.evolution_data
    validation_dir = args.validation_data if args.validation_data else args.evolution_data
    perturbation_heuristic_file = args.perturbation_heuristic
    perturbation_ratio = args.perturbation_ratio
    perturbation_time = args.perturbation_time
    max_refinement_round = args.max_refinement_round
    filter_num = args.filter_num
    evolution_rounds = args.evolution_rounds
    smoke_test = args.smoke_test
    llm_config_file = args.llm_config_file

    evolution_dir = search_file(evolution_dir, problem)
    validation_dir = search_file(validation_dir, problem)
    if not basic_heuristic_file.endswith(".py"):
        basic_heuristic_file += ".py"
    basic_heuristic_file = search_file(basic_heuristic_file, problem)
    if perturbation_heuristic_file is None:
        perturbation_heuristic_file = [file_name for file_name in os.listdir(os.path.join("src", "problems", problem, "heuristics", "basic_heuristics")) if re.match( r"random_....\.py", file_name)]
        if perturbation_heuristic_file == []:
            raise Exception("No perturbation heuristics")
        perturbation_heuristic_file = os.path.join("src", "problems", problem, "heuristics", "basic_heuristics", perturbation_heuristic_file[0])
    else:
        if not perturbation_heuristic_file.endswith(".py"):
            perturbation_heuristic_file += ".py"
        perturbation_heuristic_file = search_file(perturbation_heuristic_file, problem)

    prompt_dir=os.path.join("src", "problems", "base", "prompt")
    llm_client = get_llm_client(llm_config_file, prompt_dir, None)

    heuristic_evolver = HeuristicEvolver(llm_client, problem, evolution_dir, validation_dir)
    
    bins = tuple(int(x) for x in args.me_bins.split(","))
    bd_names = tuple(x.strip() for x in args.me_bd.split(","))
    bd_ranges = tuple(tuple(int(a) for a in r.split(":")) for r in args.me_range.split(","))

    if args.algo == "map-elites":
        from src.pipeline.ME_evolver import MapElitesEvolver

        bins = tuple(int(x) for x in args.me_bins.split(","))
        bd_names = tuple(x.strip() for x in args.me_bd.split(","))
        bd_ranges = tuple(tuple(int(a) for a in r.split(":")) for r in args.me_range.split(","))

        map_elites = MapElitesEvolver(
            llm_client=llm_client,
            heuristic_evolver=heuristic_evolver,
            problem=problem,
            evolution_dir=evolution_dir,
            validation_dir=validation_dir,
            bd_names=bd_names,
            bd_bins=bins,
            bd_ranges=bd_ranges
        )

        archive, archive_path = map_elites.run(
            seed_heuristic_file=basic_heuristic_file,
            perturbation_heuristic_file=perturbation_heuristic_file,
            perturbation_ratio=perturbation_ratio,
            perturbation_time=perturbation_time,
            max_refinement_round=max_refinement_round,
            filtered_num=filter_num,
            init_offspring=args.me_init,
            iterations=args.me_iterations,
            batch_size=args.me_batch,
            smoke_test=smoke_test,
            evolution_round=evolution_rounds
        )
        print(f"MAP-Elites done. Archive saved to: {archive_path}. Cells: {len(archive)}")
        return
    
    
    evolved_heuristics = heuristic_evolver.evolve(
        basic_heuristic_file,
        perturbation_heuristic_file,
        perturbation_ratio=perturbation_ratio,
        perturbation_time=perturbation_time,
        max_refinement_round=max_refinement_round,
        filtered_num=filter_num,
        evolution_round=evolution_rounds,
        smoke_test=smoke_test
    )
    print(evolved_heuristics)

if __name__ == "__main__":
    main()
