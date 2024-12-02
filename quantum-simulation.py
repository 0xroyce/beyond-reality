from __future__ import annotations
from abc import ABC, abstractmethod
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import numpy as np
from scipy.stats import entropy, kstest, chisquare, combine_pvalues
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class PhysicalConstants:
    """physical constants with uncertainties"""
    h: Tuple[float, float] = (6.62607015e-34, 1e-42)    # planck constant
    c: Tuple[float, float] = (299792458, 0)             # speed of light
    G: Tuple[float, float] = (6.67430e-11, 1e-15)       # gravitational constant
    alpha: Tuple[float, float] = (0.0072973525693, 1e-12)  # fine structure


class Analyzer(ABC):
    @abstractmethod
    def analyze(self) -> Dict[str, float]:
        pass


class QuantumAnalyzer(Analyzer):
    def __init__(self, samples: int = int(1e5)):
        self.samples = samples
        
    def analyze(self) -> Dict[str, float]:
        measurements = self._generate_measurements()
        return {
            'bell_violation': self._bell_test(),
            'quantum_entropy': self._calculate_entropy(measurements),
            'randomness_score': self._test_randomness(measurements),
            'entanglement_score': self._test_entanglement()
        }

    def _generate_measurements(self) -> np.ndarray:
        return np.random.normal(0, 1, self.samples)

    def _calculate_entropy(self, data: np.ndarray) -> float:
        return entropy(np.histogram(data, bins='auto')[0])

    def _bell_test(self) -> float:
        # Simplified Bell test simulation
        measurements = np.random.choice([-1, 1], size=(self.samples, 2))
        correlation = np.mean(measurements[:, 0] * measurements[:, 1])
        return abs(correlation * 2 * np.sqrt(2))

    def _test_randomness(self, data: np.ndarray) -> float:
        # Test for true randomness using KS test
        _, p_value = kstest(data, 'norm')
        return p_value

    def _test_entanglement(self) -> float:
        # simulate entangled particle pairs
        particles = np.random.choice([-1, 1], size=(self.samples, 2))
        correlation = np.abs(np.corrcoef(particles.T)[0, 1])
        return correlation


class PhysicalAnalyzer(Analyzer):
    """Physical constants and laws analysis"""

    def __init__(self, constants: PhysicalConstants):
        self.constants = constants

    def analyze(self) -> Dict[str, float]:
        # Analyze relationships between constants
        constant_values = np.array([
            self.constants.h[0],
            self.constants.c[0],
            self.constants.G[0],
            self.constants.alpha[0]
        ])

        # Reshape ratios to 2D array for correlation analysis
        ratios = np.array([
            [constant_values[i] / constant_values[j] for j in range(len(constant_values))]
            for i in range(len(constant_values))
        ])

        return {
            'ratio_entropy': entropy(np.histogram(np.log(ratios.flatten()), bins='auto')[0]),
            'correlation': self._analyze_correlations(ratios),
            'precision_pattern': self._check_precision_patterns(constant_values)
        }

    def _analyze_correlations(self, ratios: np.ndarray) -> float:
        correlation_matrix = np.corrcoef(ratios)
        mask = ~np.eye(correlation_matrix.shape[0], dtype=bool)
        return np.mean(np.abs(correlation_matrix[mask]))

    def _check_precision_patterns(self, values: np.ndarray) -> float:
        formatted = [f"{v:.8e}" for v in np.abs(values)]
        mantissas = [float(x.split('e')[0]) for x in formatted]
        return entropy(np.histogram(mantissas, bins='auto')[0])


class InformationAnalyzer(Analyzer):
    """Information theory-based detection"""

    def __init__(self, samples: int = int(1e5)):
        self.samples = samples

    def analyze(self) -> Dict[str, float]:
        data = self._generate_test_data()
        return {
            'complexity': self._calculate_complexity(data),
            'pattern_score': self._detect_patterns(data),
            'information_density': self._measure_information_density(data)
        }

    def _generate_test_data(self) -> np.ndarray:
        return np.random.random(size=self.samples)

    def _calculate_complexity(self, data: np.ndarray) -> float:
        return entropy(np.histogram(data, bins='auto')[0])

    def _detect_patterns(self, data: np.ndarray) -> float:
        fft = np.fft.fft(data)
        return entropy(np.abs(fft))

    def _measure_information_density(self, data: np.ndarray) -> float:
        hist, _ = np.histogram(data, bins='auto')
        return entropy(hist) / len(data)


@dataclass
class UserInput:
    quantum_oddities: float
    deja_vu: float
    synchronicity: float
    time_anomalies: float

    @classmethod
    def from_user_prompts(cls) -> UserInput:
        prompts = {
            'quantum_oddities': "Unexplainable glitches in daily life (past month)",
            'deja_vu': "Strong deja vu experiences (past month, 0=never, 5=weekly, 10=daily)",
            'synchronicity': "Meaningful coincidences (past week, 0=none, 5=few, 10=many)",
            'time_anomalies': "Time perception anomalies (past month)",
        }
        inputs = {}
        for key, prompt in prompts.items():
            inputs[key] = cls._get_validated_input(prompt) / 10.0
        return cls(**inputs)

    @staticmethod
    def _get_validated_input(prompt: str) -> float:
        while True:
            try:
                value = float(input(f"{prompt} (0-10): "))
                if 0 <= value <= 10:
                    return value
                print("Please enter a number between 0 and 10")
            except ValueError:
                print("Please enter a valid number")


class SimulationDetector:
    def __init__(self, samples: int = int(1e6)):
        self.analyzers = {
            'quantum': QuantumAnalyzer(samples),
            'physical': PhysicalAnalyzer(PhysicalConstants()),
            'information': InformationAnalyzer(samples)
        }
        self.config = {
            'confidence_threshold': 0.95,
            'user_weight': 0.4,
            'algo_weight': 0.6
        }

    def analyze_reality(self) -> Dict[str, str]:
        try:
            user_input = UserInput.from_user_prompts()
            detection_results = self._gather_evidence()
            probability = self._calculate_final_probability(detection_results, user_input)
            
            return self._format_results(probability, detection_results)
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise

    def _gather_evidence(self) -> Dict[str, float]:
        return {name: analyzer.analyze() 
                for name, analyzer in self.analyzers.items()}

    def _calculate_final_probability(self, evidence: Dict[str, Dict[str, float]], user_input: UserInput) -> float:
        # Normalize evidence scores
        analyzer_scores = []
        for analyzer_results in evidence.values():
            valid_scores = [v for v in analyzer_results.values() if not np.isnan(v) and v != 0]
            if valid_scores:
                analyzer_scores.append(np.mean(valid_scores))

        # Calculate algorithm probability
        algo_prob = np.mean(analyzer_scores) if analyzer_scores else 0.5

        # Calculate user probability with normalization
        user_scores = [
            user_input.quantum_oddities,
            user_input.deja_vu,
            user_input.synchronicity,
            user_input.time_anomalies
        ]
        user_prob = np.mean(user_scores)

        # Combined weighted probability
        combined_probability = (
                algo_prob * self.config['algo_weight'] +
                user_prob * self.config['user_weight']
        )

        return float(np.clip(combined_probability, 0, 1))

    def _format_results(self, probability: float, evidence: Dict[str, Dict[str, float]]) -> Dict[str, str]:
        # Calculate normalized scores
        all_scores = []
        for analyzer_results in evidence.values():
            valid_scores = [v for v in analyzer_results.values() if not np.isnan(v) and v != 0]
            if valid_scores:
                normalized = (valid_scores - np.mean(valid_scores)) / (np.std(valid_scores) + 1e-10)
                all_scores.extend(normalized)

        # Calculate metrics with safeguards
        confidence_score = float(np.clip(1 - (np.std(all_scores) if all_scores else 1), 0, 1))
        anomaly_score = float(np.clip(np.max(np.abs(all_scores)) if all_scores else 0, 0, 3))

        return {
            'conclusion': self._get_final_conclusion(probability, confidence_score),
            'likelihood': self._interpret_probability(probability),
            'evidence_strength': self._evaluate_evidence(confidence_score, anomaly_score),
            'raw_scores': {
                'simulation_probability': probability,
                'confidence_score': confidence_score,
                'anomaly_score': anomaly_score
            }
        }

    def _quantum_glitch_probability(self, user_score: float) -> float:
        # combine bell test with user observations
        bell_score = self.analyzers['quantum']._bell_test()
        return 0.3 * bell_score + 0.7 * user_score

    def _analyze_time_anomalies(self, user_time_score: float) -> float:
        # analyze time anomalies with physical correlations
        physical_score = self.analyzers['physical']._analyze_correlations(
            np.array([[1, user_time_score], [user_time_score, 1]])
        )
        return np.tanh(user_time_score * physical_score)

    def _interpret_probability(self, prob: float) -> str:
        if prob < 0.35:
            return "unlikely to be a simulation"
        elif prob < 0.55:
            return "inconclusive evidence"
        else:
            return "likely to be a simulation"

    def _get_final_conclusion(self, prob: float, conf: float) -> str:
        if prob > 0.55 and conf > 0.9:
            return "high probability we live in a simulation"
        elif prob < 0.35 and conf > 0.9:
            return "low probability we live in a simulation"
        return "current evidence is inconclusive"

    def _evaluate_evidence(self, confidence: float, anomalies: float) -> str:
        if confidence < 0.8:
            return "weak evidence"
        elif anomalies > 1.5:  # adjusted threshold
            return "strong anomalies detected"
        elif confidence > 0.95:
            return "strong evidence"
        return "moderate evidence"


# Example usage
if __name__ == "__main__":
    detector = SimulationDetector()
    analysis = detector.analyze_reality()
    
    print("\nSimulation Reality Analysis:")
    print("-" * 30)
    print(f"Conclusion: {analysis['conclusion']}")
    print(f"Likelihood: {analysis['likelihood']}")
    print(f"Evidence Strength: {analysis['evidence_strength']}")
    print("\nDetailed Scores:")
    print(f"Simulation Probability: {analysis['raw_scores']['simulation_probability']:.4f}")
    print(f"Confidence Score: {analysis['raw_scores']['confidence_score']:.4f}")
    print(f"Anomaly Score: {analysis['raw_scores']['anomaly_score']:.4f}")