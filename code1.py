def check_swap_permutation(n, k, permutation):
    # Check if there is a continuous subsegment of length k
    for i in range(n - k + 1):
        subsegment = permutation[i:i + k]
        if sorted(subsegment) == list(range(min(subsegment), max(subsegment) + 1)):
            return "YES\n0"

    # Check if there is a continuous subsegment of length k after at most one swap
    for i in range(n - k):
        if permutation[i] > permutation[i + k]:
            # Swap the elements
            permutation[i], permutation[i + k] = permutation[i + k], permutation[i]
            subsegment = permutation[i:i + k]
            if sorted(subsegment) == list(range(min(subsegment), max(subsegment) + 1)):
                return f"YES\n1\n{i + 1} {i + k + 1}"
            else:
                # Swap back the elements
                permutation[i], permutation[i + k] = permutation[i + k], permutation[i]

    return "NO"


# Example usage
n, k = map(int, input().split())
permutation = list(map(int, input().split()))

result = check_swap_permutation(n, k, permutation)
print(result)
