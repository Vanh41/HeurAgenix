import os
import ast
import json
import random
import time
import numpy as np
from typing import Dict, Tuple, Any, List, Optional
import subprocess


class MapElitesEvolver:
    def __init__(
        self,
        llm_client,
        heuristic_evolver,
        problem: str,
        evolution_dir: str,
        validation_dir: str,
        bd_names: Tuple[str, str, str] = ("SLOC", "halstead", "cyclomatic_complexity"),
        bd_bins: Tuple[int, int, int] = (10, 10, 10),
        bd_ranges: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = ((0, 300), (0, 200), (0, 50))
    ):
        self.llm_client = llm_client
        self.heuristic_evolver = heuristic_evolver
        self.problem = problem
        self.evolution_dir = evolution_dir
        self.validation_dir = validation_dir
        self.bd_names = bd_names
        self.bd_bins = bd_bins
        self.bd_ranges = bd_ranges
        self.archive: Dict[Tuple[int, ...], Dict[str, Any]] = {}
        self.output_dir = os.path.join("output", self.problem, "map_elites")
        os.makedirs(self.output_dir, exist_ok=True)
    def run(
        self,
        seed_heuristic_file: str,
        perturbation_heuristic_file: str,
        perturbation_ratio: float,
        perturbation_time: float,
        max_refinement_round: int,
        filtered_num: int,
        init_offspring: int = 8,
        iterations: int = 20,
        batch_size: int = 8,
        smoke_test: bool = False,
        evolution_round: int = 1
    ):
        # init archive
        seed_stats = self.evaluate_heuristic(seed_heuristic_file)
        self.try_insert(seed_heuristic_file, seed_stats)
        # Initial offspring from seed
        initial = self.generate_offspring(
            parent=seed_heuristic_file,
            perturbation_heuristic_file=perturbation_heuristic_file,
            perturbation_ratio=perturbation_ratio,
            perturbation_time=perturbation_time,
            max_refinement_round=max_refinement_round,
            filtered_num=filtered_num,
            count=init_offspring,
            smoke_test=smoke_test,
            evolution_round=evolution_round
        )
        for h in initial:
            stats = self.evaluate_heuristic(h)
            self.try_insert(h, stats)

        # Iterative illumination
        for it in range(iterations):
            parents = self.sample_parents(k=batch_size)
            for p in parents:
                kids = self.generate_offspring(
                    parent=p,
                    perturbation_heuristic_file=perturbation_heuristic_file,
                    perturbation_ratio=perturbation_ratio,
                    perturbation_time=perturbation_time,
                    max_refinement_round=max_refinement_round,
                    filtered_num=filtered_num,
                    count=1,
                    smoke_test=smoke_test,
                    evolution_round=evolution_round
                )
                for h in kids:
                    stats = self.evaluate_heuristic(h)
                    self.try_insert(h, stats)
                    
            self.save_archive(it)
            
        path = self.save_archive(iterations, final=True)
        return self.archive, path

    def sample_parents(self, k: int) -> List[str]:
        if not self.archive:
            return []
        cells = list(self.archive.values())
        return [random.choice(cells)["heuristic_path"] for _ in range(k)]

    def try_insert(self, heuristic_path: str, eval_stats: Optional[Dict[str, Any]]):
        if eval_stats is None:
            print(f"Evaluation failed for {os.path.basename(heuristic_path)}. Skipping.")
            return
        fitness = eval_stats["fitness"]
        bd = self.compute_bd(heuristic_path)
        cell = self.discretize(bd)
        
        curr = self.archive.get(cell)
        if (curr is None) or (fitness < curr["fitness"]): # Assuming higher fitness is better
            if curr is None:
                print(f"New cell {cell} found! Fitness: {fitness:.4f}, BD: {bd}")
            else:
                print(f"Improvement in cell {cell}! Fitness: {curr['fitness']:.4f} -> {fitness:.4f}")
            
            self.archive[cell] = {
                "heuristic_path": heuristic_path,
                "fitness": fitness,
                "bd": bd,
                "eval": eval_stats
            }

    def compute_bd(self, heuristic_path: str) -> Tuple[float, ...]:
        bd_values = []
        bd_script_dir = os.path.join("src", "pipeline", "behavior_descriptor")
        for bd_name in self.bd_names:
            script_path = os.path.join(bd_script_dir, f"{bd_name}.py")
            if not os.path.exists(script_path):
                print(f"Behavior descriptor script not found: {script_path}")
                continue
            
            try:
                result = subprocess.run(
                    ['python', script_path, heuristic_path],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=10 
                )
                value = float(result.stdout.strip())
                bd_values.append(value)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError) as e:
                print(f"Error running BD script {script_path}: {e}")
                return None 

        return tuple(bd_values)

    def discretize(self, bd: Tuple[float, ...]) -> Tuple[int, ...]:
        idx = []
        for i in range(len(bd)):
            lo, hi = self.bd_ranges[i]
            v = max(lo, min(hi, bd[i]))
            ratio = (v - lo) / (hi - lo + 1e-9)
            bin_idx = int(ratio * (self.bd_bins[i] - 1))
            idx.append(bin_idx)
        return tuple(idx)
    
    def generate_offspring(
        self,
        parent: str,
        perturbation_heuristic_file: str,
        perturbation_ratio: float,
        perturbation_time: float,
        max_refinement_round: int,
        filtered_num: int,
        count: int,
        smoke_test: bool,
        evolution_round: int
    ) -> List[str]:
        """
        Generate offspring using HeuristicEvolver
        """
        offspring = self.heuristic_evolver.evolve(
            parent,
            perturbation_heuristic_file,
            perturbation_ratio=perturbation_ratio,
            perturbation_time=perturbation_time,
            max_refinement_round=max_refinement_round,
            filtered_num=filtered_num,
            evolution_round=evolution_round,
            smoke_test=smoke_test
        )
        heuristic_paths = [item[0] for item in offspring if isinstance(item, tuple) and len(item) > 0]
        return heuristic_paths[:count] if heuristic_paths else []

    def save_archive(self, iteration: int, final: bool = False) -> str:
        payload = {
            "problem": self.problem,
            "bd_names": self.bd_names,
            "bd_bins": self.bd_bins,
            "bd_ranges": self.bd_ranges,
            "cells": [
                {
                    "cell": list(k),
                    "heuristic_path": v["heuristic_path"],
                    "fitness": v["fitness"],
                    "bd": v["bd"],
                    "eval": v["eval"]
                }
                for k, v in self.archive.items()
            ]
        }
        fname = f"archive_final.json" if final else f"archive_it_{iteration:03d}.json"
        path = os.path.join(self.output_dir, fname)
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        return path


    
    def evaluate_heuristic(self, heuristic_file: str) -> Optional[Dict[str, Any]]:
        try:
            validation_results = self.heuristic_evolver.validation(self.heuristic_evolver.validation_cases, heuristic_file)
            valid_results = [r for r in validation_results if r is not None]
            if not valid_results:
                return None  
            # Fitness is the average performance on validation cases.
            fitness = sum(valid_results) / len(valid_results)
            return {
                "fitness": fitness,
                "validation_results": validation_results  
            }
        except Exception as e:
            print(f"Error evaluating heuristic {heuristic_file}: {e}")
            return None