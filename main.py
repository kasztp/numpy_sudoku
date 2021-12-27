import argparse
from collections import Counter
from copy import deepcopy
from math import sqrt
import sys
from time import time
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("input_file",
                    help="Sudoku input file to solve (CSV).")
parser.add_argument("-v", "--verbose", action="store_true",
                    help="increase output verbosity")
args = parser.parse_args()
FILENAME = args.input_file


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


def create_mask(board: np.ndarray, dimensions: tuple[int, int]) -> list[list[int]]:
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


def update_mask(board: np.ndarray, mask: list[list[int]], box_size: int) -> list[list[int]]:
    """ Function to update Mask of possible valid values. """

    def is_list(item):
        return isinstance(item, list)

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


def update_board(board: np.ndarray, mask: list[list[int]]) -> (np.ndarray, [list[int]]):
    """ Function to update Board based on possible values Mask. """

    def is_one_element_list(item):
        return bool(isinstance(item, list) and len(item) == 1)

    for y_pos, row in enumerate(mask):
        for number in filter(is_one_element_list, row):
            x_pos = row.index(number)
            num = number.pop()
            board[y_pos][x_pos] = num
    return board, mask


def preprocess_board(board: np.ndarray, box_size: int) -> (np.ndarray, [list[int]]):
    """ Board preprocessor to reduce necessary iterations during solving. """

    mask = create_mask(board, dimensions)
    temp_mask = None
    passes = 0

    while temp_mask != mask:
        passes += 1
        temp_mask = deepcopy(mask)
        mask = update_mask(board, mask, box_size)
        board, mask = update_board(board, mask)
        # temp_mask = update_mask(board, mask, box_size)
    if args.verbose:
        print(f'Preprocess passes: {passes}')

    return np.array(board), mask


def solve(board: np.ndarray, mask, size: int, box_size: int, dimensions: tuple[int, int]) -> bool:
    """ Function to solve Sudoku with backtracking. """
    solve.iterations += 1

    find = find_empty(board)
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


def valid(board: np.ndarray, number: int, pos: tuple[int, int], box_size: int) -> bool:
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
    box = np.where(board[(box_y * box_size):(box_y * box_size + box_size),
                   (box_x * box_size):(box_x * box_size + box_size)] == number)
    location = list(zip(box[0], box[1]))

    if len(location) > 0:
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


def print_mask(mask_to_print: np.ndarray, box_size: int) -> None:
    """ Pretty print Mask of possible valid values. """
    for i, row in enumerate(mask_to_print):
        if i % box_size == 0 and i != 0:
            print('- - - - - - - - - - - -')
        print(row)


def find_empty(board: np.ndarray) -> tuple[int, int] or None:
    """ Find empty location to be filled in Sudoku. """
    location = np.argwhere(board == 0)
    if np.any(location):
        return location[0][0], location[0][1]  # row, column

    return None


if __name__ == '__main__':
    sudoku, size, dimensions = load_from_csv(FILENAME)
    box_size = int(sqrt(size))
    solve.iterations = 0

    if args.verbose:
        print_board(sudoku, size, box_size)

    start_time = time()
    sudoku, mask = preprocess_board(sudoku, box_size)
    solve(sudoku, mask, size, box_size, dimensions)
    end_time = time()

    print('_______________________\n')
    print_board(sudoku, size, box_size)
    print('_______________________\n')
    if args.verbose:
        print(f'Iterations: {solve.iterations}\n')
    print(f'Time to solve: {round(end_time - start_time, 6)}')
