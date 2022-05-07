import argparse
from collections import Counter
from copy import deepcopy
from math import sqrt
import sys
from time import time
from typing import List, Tuple
import numpy as np
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("input_file",
                    help="Sudoku input file to solve (CSV).")
parser.add_argument("-v", "--verbose", action="store_true",
                    help="increase output verbosity")
args = parser.parse_args()
FILENAME = args.input_file

size = 9
box_size = 3
dimensions = (1, 10)


def load_from_csv(filename):
    """ Function to load input from .csv, and perform basic input checking. """
    parsed = np.loadtxt(filename, delimiter=",", dtype=np.uint8)
    print('Board loaded successfully.')
    size = len(parsed)

    if size not in (9, 16) or parsed.shape[0] != parsed.shape[1]:
        print('Unsupported board size detected, exiting. (9x9 or 16x16 are supported as of now.)')
        sys.exit(1)
    if size == 16:
        start = 1
        end = 17
    else:
        start = 1
        end = 10
    dimensions = (start, end)

    if args.verbose:
        print(f'Board size detected: {size}x{size}')
    return parsed, size, dimensions


def parse_sudoku_string(data):
    """ Helper function to parse sudoku challenge. """
    size = int(sqrt(len(data)))
    box_size = int(sqrt(size))
    dimensions = (1, size + 1)
    board = []
    row = []
    for i, item in enumerate(data):
        row.append(int(item))
        if (i + 1) % size == 0:
            board.append(row)
            row = []
    return np.array(board)


def load_from_dataset(filename):
    with open(filename) as datafile:
        dataset_length = int(datafile.readline().strip())
    dataset = np.loadtxt(filename, skiprows=1, dtype=str)
    if len(dataset) == dataset_length:
        print(f'Boards loaded successfully: {dataset_length}')
    return dataset


def create_mask(board: np.ndarray, dimensions: Tuple[int, int]) -> List[List[int]]:
    """ Function to create Mask of possible valid values based on the initial sudoku Board. """

    mask = list(board.tolist())
    counts = Counter(board.flatten())
    del counts[0]
    counts = [number[0] for number in counts.most_common()]
    most_common_clues = counts
    for clue in range(dimensions[0], dimensions[1]):
        if clue not in most_common_clues:
            most_common_clues.append(clue)

    for i, row in enumerate(mask):
        if 0 in row:
            while 0 in row:
                zero_index = row.index(0)
                mask[i][zero_index] = []
                for number in most_common_clues:
                    if valid(board, number, (i, zero_index), box_size):
                        mask[i][zero_index].append(number)
        else:
            for number in row:
                if number != 0:
                    mask[i][row.index(number)] = {number}
    return mask


def update_mask(board: np.ndarray, mask: List[List[int]], box_size: int) -> List[List[int]]:
    """ Function to update Mask of possible valid values. """

    def is_list(item):
        return bool(isinstance(item, list))

    for y_pos, row in enumerate(mask):
        for numbers in filter(is_list, row):
            x_pos = row.index(numbers)
            to_remove = set()
            for number in numbers:
                if not valid(board, number, (y_pos, x_pos), box_size):
                    to_remove.add(number)
            for num in to_remove:
                mask[y_pos][x_pos].remove(num)
    return mask


def update_board(board: np.ndarray, mask: List[List[int]]) -> (np.ndarray, [List[int]]):
    """ Function to update Board based on possible values Mask. """

    def is_one_element_list(item):
        return bool(isinstance(item, list) and len(item) == 1)

    for y_pos, row in enumerate(mask):
        for number in filter(is_one_element_list, row):
            x_pos = row.index(number)
            num = number.pop()
            board[y_pos][x_pos] = num
    return board, mask


def preprocess_board(board: np.ndarray, box_size: int) -> (np.ndarray, [List[int]]):
    """ Board preprocessor to reduce necessary iterations during solving. """

    mask = create_mask(board, dimensions)
    temp_mask = None
    passes = 0

    while temp_mask != mask:
        passes += 1
        temp_mask = deepcopy(mask)
        mask = update_mask(board, mask, box_size)
        board, mask = update_board(board, mask)

    if args.verbose:
        print(f'Preprocess passes: {passes}')

    return np.array(board), mask


def solve(board: np.ndarray, mask, size: int, box_size: int, dimensions: Tuple[int, int]) -> bool:
    """ Function to solve Sudoku with backtracking. """
    #solve.iterations += 1

    find = find_min_empty(board, mask)
    # find = find_empty(board)  # Old method for finding empty cells.
    if not find:
        return True
    row, col = find

    for number in mask[row][col]:
        if valid(board, number, (row, col), box_size):
            board[row, col] = number
            if solve(board, mask, size, box_size, dimensions):
                return True
            board[row, col] = 0

    return False


def valid(board: np.ndarray, number: int, pos: Tuple[int, int], box_size: int) -> bool:
    """ Function to check if a given value is valid for the specific location of the sudoku. """
    # Check row
    location = np.where(board[pos[0]] == number)
    if len(location[0]):
        return False

    # Check column
    location = np.where(board[:, pos[1]] == number)
    if len(location[0]):
        return False

    # Check box
    box_x = pos[1] // box_size
    box_y = pos[0] // box_size
    location = np.where(board[(box_y * box_size):(box_y * box_size + box_size),
                   (box_x * box_size):(box_x * box_size + box_size)] == number)

    if len(location[0]):
        return False
    return True


def print_board(board: np.ndarray, size: int, box_size: int) -> None:
    """ Pretty print Sudoku board. """
    for i, row in enumerate(board):
        if i % box_size == 0 and i != 0:
            print('- - - - - - - - - - - -')

        for j, _ in enumerate(row):
            if j % box_size == 0 and j != 0:
                print(' | ', end='')

            if j == (size - 1):
                print(board[i, j])
            else:
                print(str(board[i, j]) + ' ', end='')


def print_mask(mask_to_print: List[List[int]], box_size: int) -> None:
    """ Pretty print Mask of possible valid values. """
    for i, row in enumerate(mask_to_print):
        if i % box_size == 0 and i != 0:
            print('- - - - - - - - - - - -')
        print(row)


def find_empty(board: np.ndarray) -> Tuple[int, int] or None:
    """ Find empty location to be filled in Sudoku. """
    location = np.argwhere(board == 0)
    if np.any(location):
        return location[0][0], location[0][1]  # row, column

    return None


def find_min_empty(board: np.ndarray, mask: List[List[int]]) -> Tuple[int, int] or None:
    """ Find empty location to be filled in Sudoku,
        where the number of possible values is optimal. """

    def not_zero_element_list(item):
        return bool(isinstance(item, list) and len(item) != 0)

    shortest_cue_lists = {}
    for y_pos, row in enumerate(mask):
        sorted_lists = sorted(filter(not_zero_element_list, row), key=len, reverse=True)
        if len(sorted_lists) >= 1:
            for item in sorted_lists:
                shortest_cue_lists[(y_pos, row.index(item))] = len(sorted_lists[0])
    shortest_cue_lists = dict(sorted(shortest_cue_lists.items(), key=lambda item: item[1]))

    for coordinate in shortest_cue_lists.keys():
        if board[coordinate[0]][coordinate[1]] == 0:
            return coordinate[0], coordinate[1]  # row, column

    return find_empty(board)


def solver(x):
    size = 9
    box_size = 3
    dimensions = (1, 10)
    board = parse_sudoku_string(x)
    #print(board)
    sudoku, mask = preprocess_board(board, box_size)
    #print(sudoku)
    solve(sudoku, mask, size, box_size, dimensions)
    print(sudoku)
    solution = ''
    for row in sudoku:
        for number in row:
            solution += str(number)
    return solution

vectorize = np.vectorize(solver)

if __name__ == '__main__':
    a = load_from_dataset(FILENAME)
    if args.verbose:
        print(f'Parsed input: {a}')
        print(f'Type of a: {type(a)}')
        print(f'Shape of a: {a.shape}')
    start_time = time()
    #a = vectorize(a)
    #a = np.apply_along_axis(vectorize, 0, a)
    a = np.array([solver(x) for x in a])
    end_time = time()
    print(a)
    print(f'Time to solve: {round(end_time - start_time, 6)}')
    """sudoku, size, dimensions = load_from_csv(FILENAME)
    box_size = int(sqrt(size))
    solve.iterations = 0

    if args.verbose:
        print_board(sudoku, size, box_size)

    start_time = time()
    sudoku, mask = preprocess_board(sudoku, box_size)

    if args.verbose:
        print('Preprocessed:')
        print_board(sudoku, size, box_size)

    solve(sudoku, mask, size, box_size, dimensions)
    end_time = time()

    print('_______________________\n')
    print_board(sudoku, size, box_size)
    print('_______________________\n')
    if args.verbose:
        print(f'Iterations: {solve.iterations}\n')
    print(f'Time to solve: {round(end_time - start_time, 6)}')
"""