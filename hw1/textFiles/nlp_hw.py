import numpy as np
from enum import IntEnum

class Score(IntEnum):
    match = 2
    mismatch = -1
    gap_cost = -1

def smith_waterman_algorithm(line1, line2):
    matrix_result = np.zeros((len(line1) + 1, len(line2) + 1), int)
    for i in range(1,len(matrix_result)): 
        for j in range(1,len(matrix_result[0])):
            if line1[i - 1] == line2[j - 1]:
                 substitution = matrix_result[i - 1, j - 1] + Score.match
            else:
                substitution = matrix_result[i - 1, j - 1] + Score.mismatch
            delete = matrix_result[i - 1, j] + Score.gap_cost
            insert = matrix_result[i, j - 1] + Score.gap_cost
            matrix_result[i, j] = max(substitution, delete, insert, 0)

    return matrix_result


def traceback(matrix_result, line1, line2, alignment1='',alignment2=''):
    max_point=0
    max_i = 0
    max_j =0
    first_control = 0
    for i in range(1,len(matrix_result)): 
        for j in range(1,len(matrix_result[0])):
            if matrix_result[i][j] > max_point or matrix_result[i][j] == max_point :
                max_point = matrix_result[i][j]
                max_i = i
                max_j = j

    while matrix_result[max_i, max_j] != 0:
        up = matrix_result[max_i-1, max_j]
        left = matrix_result[max_i, max_j-1]
        diagonal = matrix_result[max_i-1, max_j-1]
        best_way = max(up,left,diagonal)

        if best_way == 0:
            alignment1 = alignment1[::-1]
            alignment2 = alignment2[::-1]
            return alignment1,alignment2

        if first_control == 0: 
            alignment1 = alignment1 + line1[max_i-1]
            alignment2 = alignment2 + line2[max_j-1]

        if best_way == diagonal:
            if line1[max_i-2] == line2[max_j-2]:
                alignment1 = alignment1 + line1[max_i-2]
                alignment2 = alignment2 + line2[max_j-2]
                max_i = max_i - 1
                max_j = max_j - 1
            else:
                alignment1 = alignment1 + line1[max_i-2]
                alignment2 = alignment2 + '-'
                max_i = max_i - 1
                max_j = max_j - 1

        elif best_way == up:
            alignment1 = alignment1 + '-'
            max_i = max_i - 1
        
        elif best_way == left:
            alignment2 = alignment2 + '-'
            max_j = max_j - 1
        first_control = 1 

    alignment1 = alignment1[::-1]
    alignment2 = alignment2[::-1]
    return alignment1,alignment2
    


def read_file():
    fileName1 = input("Enter name of file1 : ")
    fileName2 = input("Enter name of file2 : ")

    file1 = open(fileName1, 'r', encoding="utf8")
    Lines = file1.readlines()
    file2 = open(fileName2, 'r', encoding="utf8")
    Lines2 = file2.readlines()

    for line1 in Lines:
        for line2 in Lines2:
            if len(line1)>5 and len(line2)>5:
                H = smith_waterman_algorithm(line1, line2)
                a, b = traceback(H,line1,line2)
                if line1==b:
                    print(H)
                    print("line1: "+a)
                    print("line2: "+b)
                    print("Lines are the same")
  

"""a, b = 'ATCAT', 'ATTATC' 
H = matrix(a, b)
print("Two lines to compare:")
print("line1: "+a)
print("line2: "+b)
print("\n")
a, b = traceback(H,a,b)
print(H)
print("\nTraceback Result:")
print("Alignment1: "+a)
print("Alignment2: "+b)"""

read_file()
