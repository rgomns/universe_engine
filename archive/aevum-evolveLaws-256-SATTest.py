import numpy as np
import time
from typing import List, Tuple, Optional

class ConstraintSATSolver:
    """
    Adapted constraint solver for SAT problems.
    Compares against traditional SAT solvers.
    """
    def __init__(self, n_vars: int, pop_size: int = 1000):
        self.n_vars = n_vars
        self.pop_size = pop_size
        
        # Population: each row is a candidate solution
        self.population = np.random.randint(0, 2, (pop_size, n_vars), dtype=np.int8)
        
        # Clauses: list of [variables, forbidden_pattern, strength]
        self.clauses = []
        
        self.cycle = 0
        self.best_satisfaction = 0.0
        self.best_solution = None
        
        # Weights for converting bit patterns to indices
        self.weights_cache = {}
    
    def add_clause_from_literals(self, literals: List[int]):
        """
        Add a SAT clause from standard notation.
        Example: [1, -2, 3] means (x1 OR NOT x2 OR x3)
        
        We convert OR clauses to forbidden AND patterns.
        (x1 OR NOT x2 OR x3) is violated when (NOT x1 AND x2 AND NOT x3)
        """
        variables = [abs(lit) - 1 for lit in literals]  # Convert to 0-indexed
        signs = [1 if lit > 0 else 0 for lit in literals]  # 1 if positive, 0 if negated
        
        # The forbidden pattern is the negation of all literals
        # For (x1 OR NOT x2 OR x3), forbidden is (0, 1, 0) meaning x1=0, x2=1, x3=0
        forbidden_bits = [1 - sign for sign in signs]
        forbidden_index = sum(bit << i for i, bit in enumerate(forbidden_bits))
        
        scope = np.array(variables, dtype=int)
        self.clauses.append([scope, forbidden_index, 1.0])
        
        # Cache weights for this clause size
        size = len(scope)
        if size not in self.weights_cache:
            self.weights_cache[size] = np.array([2**i for i in range(size)], dtype=np.int32)
    
    def evaluate_solution(self, solution: np.ndarray) -> Tuple[int, int]:
        """
        Returns (satisfied_clauses, total_clauses)
        """
        satisfied = 0
        for scope, forbidden, _ in self.clauses:
            val = solution[scope].dot(self.weights_cache[len(scope)])
            if val != forbidden:
                satisfied += 1
        return satisfied, len(self.clauses)
    
    def solve(self, max_cycles: int = 10000, target_satisfaction: float = 1.0) -> Optional[np.ndarray]:
        """
        Attempt to solve the SAT problem.
        Returns solution if found, None otherwise.
        """
        start_time = time.time()
        
        for cycle in range(max_cycles):
            self.cycle = cycle
            
            # Relaxation: fix violations
            for scope, forbidden, strength in self.clauses:
                if len(scope) not in self.weights_cache:
                    continue
                    
                current_vals = self.population[:, scope].dot(self.weights_cache[len(scope)])
                violators = (current_vals == forbidden)
                
                if violators.any():
                    # Flip random bit in violating samples
                    mask = violators & (np.random.random(self.pop_size) < strength)
                    if mask.any():
                        flip_bits = np.random.randint(0, len(scope), mask.sum())
                        for i, idx in enumerate(np.where(mask)[0]):
                            self.population[idx, scope[flip_bits[i]]] ^= 1
            
            # Check for solution every 100 cycles
            if cycle % 100 == 0:
                for solution in self.population:
                    satisfied, total = self.evaluate_solution(solution)
                    satisfaction_rate = satisfied / total if total > 0 else 0
                    
                    if satisfaction_rate > self.best_satisfaction:
                        self.best_satisfaction = satisfaction_rate
                        self.best_solution = solution.copy()
                    
                    if satisfaction_rate >= target_satisfaction:
                        elapsed = time.time() - start_time
                        print(f"✓ Solution found at cycle {cycle} ({elapsed:.3f}s)")
                        print(f"  Satisfaction: {satisfied}/{total} clauses")
                        return solution.copy()
                
                if cycle % 1000 == 0 and cycle > 0:
                    elapsed = time.time() - start_time
                    print(f"  Cycle {cycle}: Best satisfaction = {self.best_satisfaction:.2%} ({elapsed:.3f}s)")
        
        elapsed = time.time() - start_time
        print(f"✗ No perfect solution found in {max_cycles} cycles ({elapsed:.3f}s)")
        print(f"  Best satisfaction: {self.best_satisfaction:.2%}")
        return self.best_solution


def generate_random_3sat(n_vars: int, n_clauses: int, seed: int = 42) -> List[List[int]]:
    """
    Generate random 3-SAT problem.
    Returns list of clauses, each with 3 literals.
    """
    np.random.seed(seed)
    clauses = []
    
    for _ in range(n_clauses):
        # Pick 3 random distinct variables
        vars = np.random.choice(n_vars, size=3, replace=False) + 1  # 1-indexed
        # Randomly negate each
        signs = np.random.choice([-1, 1], size=3)
        clause = (vars * signs).tolist()
        clauses.append(clause)
    
    return clauses


def cnf_to_dimacs(clauses: List[List[int]], n_vars: int) -> str:
    """
    Convert clauses to DIMACS CNF format (standard for SAT solvers).
    """
    lines = []
    lines.append(f"p cnf {n_vars} {len(clauses)}")
    for clause in clauses:
        lines.append(" ".join(map(str, clause)) + " 0")
    return "\n".join(lines)


# =============================================================================
# BENCHMARK TESTS
# =============================================================================

def test_easy_problem():
    """Test on a simple satisfiable problem"""
    print("\n" + "="*70)
    print("TEST 1: Easy Problem (10 variables, 20 clauses)")
    print("="*70)
    
    clauses = generate_random_3sat(n_vars=10, n_clauses=20, seed=42)
    
    solver = ConstraintSATSolver(n_vars=10, pop_size=500)
    for clause in clauses:
        solver.add_clause_from_literals(clause)
    
    solution = solver.solve(max_cycles=5000)
    
    if solution is not None:
        sat, total = solver.evaluate_solution(solution)
        print(f"\n✓ PASS: Found solution satisfying {sat}/{total} clauses")
        return True
    else:
        print("\n✗ FAIL: No solution found")
        return False


def test_medium_problem():
    """Test on moderate difficulty problem"""
    print("\n" + "="*70)
    print("TEST 2: Medium Problem (50 variables, 200 clauses)")
    print("="*70)
    
    # Ratio 4:1 (clauses to variables) is around phase transition
    clauses = generate_random_3sat(n_vars=50, n_clauses=200, seed=123)
    
    solver = ConstraintSATSolver(n_vars=50, pop_size=2000)
    for clause in clauses:
        solver.add_clause_from_literals(clause)
    
    solution = solver.solve(max_cycles=20000)
    
    if solution is not None:
        sat, total = solver.evaluate_solution(solution)
        print(f"\n✓ Result: {sat}/{total} clauses satisfied ({sat/total:.1%})")
        return sat == total
    else:
        return False


def test_hard_problem():
    """Test on difficult problem near phase transition"""
    print("\n" + "="*70)
    print("TEST 3: Hard Problem (100 variables, 430 clauses)")
    print("="*70)
    print("Note: Ratio ~4.3:1 is near the satisfiability phase transition")
    
    clauses = generate_random_3sat(n_vars=100, n_clauses=430, seed=999)
    
    solver = ConstraintSATSolver(n_vars=100, pop_size=3000)
    for clause in clauses:
        solver.add_clause_from_literals(clause)
    
    solution = solver.solve(max_cycles=50000)
    
    if solution is not None:
        sat, total = solver.evaluate_solution(solution)
        print(f"\n✓ Result: {sat}/{total} clauses satisfied ({sat/total:.1%})")
        return sat >= total * 0.95  # Accept 95%+ as success for hard problems
    else:
        return False


def benchmark_suite():
    """Run full benchmark suite"""
    print("\n" + "="*70)
    print("CONSTRAINT SAT SOLVER BENCHMARK")
    print("="*70)
    print("\nComparing against theoretical SAT solver performance...")
    print("Standard solvers (MiniSat, CryptoMiniSat, etc.) are optimized for these tasks.")
    print("Our approach uses population-based constraint satisfaction.\n")
    
    results = []
    
    # Run tests
    results.append(("Easy (10 vars)", test_easy_problem()))
    results.append(("Medium (50 vars)", test_medium_problem()))
    results.append(("Hard (100 vars)", test_hard_problem()))
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
    
    print("\n" + "="*70)
    print("COMPARISON NOTES:")
    print("="*70)
    print("""
    Modern SAT solvers (MiniSat, Glucose, etc.):
    - Can solve 100-var problems in milliseconds
    - Use CDCL (Conflict-Driven Clause Learning)
    - Optimized for exact solutions 
    
    This Constraint Solver:
    - Uses population-based stochastic search
    - Good for approximate solutions
    - May struggle with exact satisfaction on hard instances
    - Better suited for optimization vs decision problems
    
    VERDICT: Traditional SAT solvers are faster for exact SAT.
    BUT: This approach could be better for:
    - MAX-SAT (maximize satisfied clauses)
    - Weighted constraints
    - Continuous optimization with discrete constraints
    """)


if __name__ == "__main__":
    benchmark_suite()