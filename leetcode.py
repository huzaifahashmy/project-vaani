class Solution:
    def smallestPalindrome(self, s: str) -> str:
        length = len(s)
        if len ==1 :
            return s
        
        mid = length //2
        if len(s)%2 == 0:
            left_value = "".join(sorted(s[:mid]))
            righ_value = "".join(sorted(s[mid:],reverse=True))
            
            return left_value + righ_value
        else:
            left_value = "".join(sorted(s[:mid]))
            righ_value = "".join(sorted(s[mid+1:],reverse=True))
            
            return left_value + s[mid] + righ_value
        



        
# text = "Python Programming"
# limit = 6

# # Slice up to the limit, reverse it, then add the rest of the string
# reversed_text = text[:limit][::-1] + text[limit:]


# s = "edcab"
# sorted_string = "".join(sorted(s))
# print(sorted_string)  # Output: "abcde"

print(Solution().smallestPalindrome(s = "yey"))

