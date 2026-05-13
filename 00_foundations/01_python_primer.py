"""
Module 0 — Python primer for this project
==========================================

Just the bits of Python you'll see throughout. Run this file:
    python 01_python_primer.py

Read each section, predict the output, then run it.
"""

# ---------------------------------------------------------------
# 1. Variables and basic types
# ---------------------------------------------------------------
x = 5                     # integer
y = 3.14                  # float
name = "Claude"           # string
is_smart = True           # boolean
print("1. Basic types:", x, y, name, is_smart)


# ---------------------------------------------------------------
# 2. Lists (ordered collections — like vectors)
# ---------------------------------------------------------------
nums = [10, 20, 30, 40]
print("2. List:", nums)
print("   first element:", nums[0])     # 10  (Python is 0-indexed)
print("   last element :", nums[-1])    # 40
print("   length       :", len(nums))   # 4


# ---------------------------------------------------------------
# 3. Dictionaries (key → value lookups — like a tokenizer!)
# ---------------------------------------------------------------
token_to_id = {"hello": 0, "world": 1, "claude": 2}
print("3. Dict:", token_to_id)
print("   id for 'claude':", token_to_id["claude"])


# ---------------------------------------------------------------
# 4. Functions
# ---------------------------------------------------------------
def square(n):
    """Return n squared."""
    return n * n

print("4. square(7) =", square(7))


# ---------------------------------------------------------------
# 5. Loops
# ---------------------------------------------------------------
total = 0
for n in nums:
    total += n
print("5. sum of nums:", total)


# ---------------------------------------------------------------
# 6. List comprehensions (compact loops — VERY common in ML code)
# ---------------------------------------------------------------
squares = [square(n) for n in [1, 2, 3, 4, 5]]
print("6. squares:", squares)


# ---------------------------------------------------------------
# 7. Classes (a blueprint for an object)
#    We'll use these for tokenizers, models, etc.
# ---------------------------------------------------------------
class Counter:
    def __init__(self):           # constructor, runs when you create an instance
        self.value = 0

    def increment(self):
        self.value += 1

c = Counter()
c.increment()
c.increment()
c.increment()
print("7. counter value:", c.value)


# ---------------------------------------------------------------
# 8. f-strings — nice formatted output
# ---------------------------------------------------------------
loss = 0.0423
print(f"8. epoch=3  loss={loss:.4f}")   # rounds to 4 decimals


# ---------------------------------------------------------------
# That's everything. You now read enough Python to follow any
# file in this project. Run me, then open 02_math_primer.py.
# ---------------------------------------------------------------
