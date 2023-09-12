import heapq


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


input = [1, 4328, 6, 8, 0, 3, 5, 6, 8, 92]

print(top_three(input))
