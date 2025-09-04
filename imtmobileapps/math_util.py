import heapq
from tkinter.constants import CENTER


# You can do it using a priority queue or a heap in Python. Here's how you can do it:
# The `heap[0]` operation always gives the smallest element.
# `heapreplace()` pop and return the smallest element,
# and push the new item on the heap, maintaining heap invariance.
# And Time complexity is O(n).

# Note: This will work even when there are less than 3 numbers in the array.
# And when there are duplicate largest numbers, it will consider the duplicates
# as separate numbers

def top_three(nums):
    """ Returns the three largest numbers in nums"""
    # Initialize a heap of size 3
    heap = [-float('inf')] * 3
    for num in nums:
        if num > heap[0]:
            # If current number is larger than the smallest in heap, replace it
            heapq.heapreplace(heap, num)
    return [heapq.heappop(heap) for _ in range(3)][::-1]


myInput = [1, 4328, 6, 8, 0, 3, 5, 6, 8, 92]

print(top_three(myInput))


# run this from the terminal - 'python math_util.py'
def decimal_to_binary(decimal):
    if decimal == 0:
        return "0"
    binary = ""
    num = abs(decimal)
    while num > 0:
        binary = str(num % 2) + binary
        num //= 2
    return binary if decimal >= 0 else "-" + binary
# enter decimal number
print(decimal_to_binary())

def binary_to_decimal(binary):
    if binary.startswith("-"):
        sign = -1
        binary = binary[1:]
    else:
        sign = 1
    decimal = 0
    for digit in binary:
        if digit not in "01":
            raise ValueError("Invalid binary string")
        decimal = decimal * 2 + int(digit)
    return sign * decimal
# enter binary as string
print(binary_to_decimal(""))
