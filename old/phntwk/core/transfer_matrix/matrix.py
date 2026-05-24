import numpy as np
from typing import Literal

def flip(matrix: np.ndarray) -> np.ndarray:
    """reverse the input and output ports sequence of a matrix"""
    return np.flip(matrix, axis=(1, 2))

def swap(matrix: np.ndarray) -> np.ndarray:
    """swap input and output ports of  matrix"""
    return np.transpose(matrix, axes=(0, 2, 1))

def rotate(matrix: np.ndarray) -> np.ndarray:
    """swap the input and output ports and reverse sequence of a matrix"""
    return swap(flip(matrix))

def transform_matrix_by_orient(matrix: np.ndarray, orient:Literal["MX","MY","R180"]):
    match orient:
        case "MX":
            return flip(matrix)
        case "MY":
            return swap(matrix)
        case "R180":
            return rotate(matrix)
        case _:
            raise ValueError(f"unsupported orient: {orient} ")

if __name__ == "__main__":
    sample_matrix = np.arange(24).reshape(4, 2, 3)
    print(f"matrix: \n{sample_matrix}")
    print(f"flip: \n{flip(sample_matrix)}")
    print(f"swap: \n{swap(sample_matrix)}")
    print(f"rotate: \n{rotate(sample_matrix)}")